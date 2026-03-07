from pathlib import Path
import pickle
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
import time
import numpy as np

from src.metrics import MR, minADE, minFDE, brier_minFDE, AvgMinADE, ActorMR, AvgMinFDE, AvgBrierMinFDE
from src.utils.optim import WarmupCosLR


class BaseLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: dict,
        optim: dict = None,
        ma = False
    ) -> None:
        super(BaseLightningModule, self).__init__()
        self.time_list = []
        self.optim = optim

        self.model = model
        self.curr_ep = 0
        self.ma = ma
        self.ma_eval_only = False
        self.reset_val_scores(init=True)      
        self.val_durations = []
        self.last_val_durations = []

        return

    def forward(self, data):
        return self.model(data)
    
    def load_chkpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("model.") :]: v for k, v in ckpt.items() if k.startswith("model.")
        }
        self.model.load_state_dict(state_dict=state_dict, strict=False)

    def reset_val_scores(self, init=False):
        if self.ma:
            if init:
                self.metrics = MetricCollection(
                    {
                        "AvgMinADE": AvgMinADE(),
                        "AvgMinFDE": AvgMinFDE(),
                        "ActorMR": ActorMR(),
                        "AvgBrierMinFDE": AvgBrierMinFDE()
                    }
                )
            self.val_scores = {"AvgMinADE": [], "AvgMinFDE": [], "ActorMR": [], "AvgBrierMinFDE": []}
        else:
            if init:
                self.metrics = MetricCollection(
                    {
                        'minADE1': minADE(k=1),
                        'minADE6': minADE(k=6),
                        'minFDE1': minFDE(k=1),
                        'minFDE6': minFDE(k=6),
                        'MR': MR(),
                        'b-minFDE6': brier_minFDE(k=6),
                    }
                )
            self.val_scores = {"MR": [], "minADE1": [], "minADE6": [], "minFDE1": [], "minFDE6": [], "b-minFDE6": []}   

    """
        Compute multi-agent loss (joint predictions).
    """
    def ma_cal_loss(self, out, data, tag=''):
        gt_len = data['target'][:, 0].shape[-2] 
        y_hat = out['y_hat'][:, :, :, :gt_len]
        y = data['target'][:, :, :y_hat.shape[3]]
        valid_mask = data["scored_mask"].unsqueeze(1)
        scene_avg_ade = (
            torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1) * valid_mask
        ).sum(dim=-1) / valid_mask.sum(dim=-1)
        best_mode = torch.argmin(scene_avg_ade, dim=-1)

        y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]

        pi = out["pi"]
        reg_mask = data["scored_mask"].unsqueeze(-1).unsqueeze(-1).expand_as(y_hat_best)
        agent_reg_loss = F.smooth_l1_loss(y_hat_best[reg_mask], y[reg_mask])
        agent_cls_loss = F.cross_entropy(pi, best_mode.detach())

        loss = agent_reg_loss + agent_cls_loss
        disp_dict = {
            f'{tag}loss': loss.item(),
            f'{tag}reg_loss': agent_reg_loss.item(),
            f'{tag}cls_loss': agent_cls_loss.item(),
        }

        return loss, disp_dict

    """
        Compute single-agent loss (marginal predictions).
    """
    def cal_loss(self, out, data, tag='', ma=False):
        if ma: return self.ma_cal_loss(out, data, tag=tag)
        gt_len = data['target'][:, 0].shape[-2]
        y_hat, pi, y_hat_others = out['y_hat'][:, :, :gt_len], out['pi'], out['y_hat_others'][:, :, :gt_len]
        new_y_hat = out.get('new_y_hat', None)
        y, y_others = data['target'][:, 0], data['target'][:, 1:]
        if new_y_hat is None:
            l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
        else:
            l2_norm = torch.norm(new_y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]

        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
        agent_cls_loss = F.cross_entropy(pi, best_mode.detach())
        if new_y_hat is not None:
            new_y_hat_best = new_y_hat[torch.arange(new_y_hat.shape[0]), best_mode]
            new_agent_reg_loss = F.smooth_l1_loss(new_y_hat_best[..., :2], y)
        else:
            new_agent_reg_loss = 0

        others_reg_mask = data['target_mask'][:, 1:]
        others_reg_loss = F.smooth_l1_loss(
            y_hat_others[others_reg_mask], y_others[others_reg_mask]
        )

        loss = agent_reg_loss + agent_cls_loss + others_reg_loss + new_agent_reg_loss
        disp_dict = {
            f'{tag}loss': loss.item(),
            f'{tag}reg_loss': agent_reg_loss.item(),
            f'{tag}cls_loss': agent_cls_loss.item(),
            f'{tag}others_reg_loss': others_reg_loss.item(),
        }
        if new_y_hat is not None:
            disp_dict[f'{tag}reg_loss_refine'] = new_agent_reg_loss.item()

        return loss, disp_dict

    def training_step(self, data, batch_idx):
        if isinstance(data, list):
            data = data[-1]
        out = self(data)
        loss, loss_dict = self.cal_loss(out, data)

        for k, v in loss_dict.items():
            self.log(
                f'train/{k}',
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=len(data["scenario_id"])
            )

        return loss
    
    def on_validation_end(self) -> None:
        # report metrics and latencies
        m = np.mean(self.val_durations)    
        ml = np.mean(self.last_val_durations)        
        score_str = " ".join( ["{}: {:5.3f}".format(k, np.mean(self.val_scores[k])) for k in self.val_scores.keys()] )
        if "val_minFDE6" in self.val_scores.keys() and "val_brier-minFDE6" in self.val_scores.keys():
            score_str += " ({:5.3f})".format( np.mean(self.val_scores["val_brier-minFDE6"]) - np.mean(self.val_scores["val_minFDE6"]) )
        print( "[{:2d}] Avg over {} batches: {}/{} ms - {}".format(self.curr_ep, len(self.val_durations), int(m*1000), int(ml*1000), score_str) )
        if self.ma:
            print( " & ".join( ["{:5.3f}".format(np.mean(self.val_scores[k])) for k in ["AvgMinADE", "AvgMinFDE", "ActorMR", "AvgBrierMinFDE"]] ) )
        else:
            print( " & ".join( ["{:5.3f}".format(np.mean(self.val_scores[k])) for k in ["MR", "minADE6", "minFDE6", "b-minFDE6"]] ) )
        self.curr_ep += 1

    def on_validation_start(self) -> None:
        self.reset_val_scores()
        self.val_durations = []
        self.last_val_durations = []

    def validation_step(self, data, batch_idx):
        if isinstance(data, list):
            data = data[-1]
        st = time.time()
        out = self(data)
        end = time.time()
        self.val_durations.append(end-st)
        self.last_val_durations.append(end-st)

        _, loss_dict = self.cal_loss(out, data)
        if self.ma:
            metrics = self.metrics(out, data['target'], data["scored_mask"])
        else:
            metrics = self.metrics(out, data['target'][:, 0])

        self.log(
            'val/reg_loss',
            loss_dict['reg_loss'],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=len(data["scenario_id"])
        )
        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(data["scenario_id"])
        )
        for k in self.val_scores.keys(): self.val_scores[k].append(metrics[k].item())
        return [out]


    def on_test_start(self) -> None:
        save_dir = Path('./submission')
        save_dir.mkdir(exist_ok=True)
        if self.ma:
            from src.utils.ma_submission_av2 import SubmissionAv2
        else:
            from src.utils.submission_av2 import SubmissionAv2
        self.submission_handler = SubmissionAv2(
            save_dir=save_dir
        )

    def test_step(self, data, batch_idx) -> None:
        if isinstance(data, list):
            data = data[-1]
        out = self(data)
        self.submission_handler.format_data(data, out['y_hat'], out['pi'])

    def on_test_end(self) -> None:
        self.submission_handler.generate_submission_file()

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    '%s.%s' % (module_name, param_name) if module_name else param_name
                )
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                'params': [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                'weight_decay': self.optim.weight_decay,
            },
            {
                'params': [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                'weight_decay': 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.optim.lr, weight_decay=self.optim.weight_decay
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.optim.lr,
            min_lr=self.optim.min_lr,
            warmup_ratio=self.optim.warmup_ratio,
            epochs=self.optim.epochs,
        )
        return [optimizer], [scheduler]


class StreamLightningModule(BaseLightningModule):
    def __init__(self,
                 num_grad_frame=3,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_grad_frame = num_grad_frame

        from .ma_seam import Seam_ma as ConsistencyModule
        if self.ma and not self.ma_eval_only: self.consistency_module = ConsistencyModule()
        return

    def training_step(self, data, batch_idx):
        total_step = len(data)
        num_grad_frames = min(self.num_grad_frame, total_step)
        num_no_grad_frames = total_step - num_grad_frames

        memory_dict = None
        self.eval()
        # iterate over prediction time steps, which do not require gradient
        with torch.no_grad():
            # marginal predictions 
            for i in range(num_no_grad_frames):
                cur_data = data[i]
                cur_data['memory_dict'] = memory_dict
                out = self(cur_data)
                memory_dict = out['memory_dict']
        
        self.train()
        sum_loss = 0
        loss_dict = {}
        # iterate over prediction time steps, which require gradient
        for i in range(num_grad_frames):
            # marginal predictions an sa loss
            cur_data = data[i + num_no_grad_frames]
            cur_data['memory_dict'] = memory_dict
            out = self(cur_data)
            cur_loss, cur_loss_dict = self.cal_loss(out, cur_data, tag=f'step{i + num_no_grad_frames}_')
            loss_dict.update(cur_loss_dict)
            memory_dict = out['memory_dict']

            if self.ma:
                # joint predictions and ma loss
                sum_loss += cur_loss
                ma_input = self.getScenario(out, cur_data)
                scene_out = self.consistency_module(ma_input)
                cur_data["scored_mask"] = ma_input["x_key_valid_mask"]
                cur_data["target"] = ma_input["target"]
                ma_loss, __ = self.cal_loss(scene_out, cur_data, tag=f'step{i + num_no_grad_frames}_', ma=True)
                sum_loss += ma_loss * 0.9 + cur_loss * 0.1
                
                # stream joint predictions
                y_hat = scene_out["y_hat"].permute(0, 2, 1, 3, 4)[ma_input["x_key_valid_mask"]]
                B = y_hat.shape[0]
                glo_y_hat = torch.bmm(y_hat.detach().reshape(B, -1, 2), torch.inverse(memory_dict["rot_mat"]))
                memory_dict["glo_y_hat"] = glo_y_hat.reshape(B, y_hat.size(1), -1, 2)
                memory_dict["x_mode"] = scene_out["x_mode"].permute(0, 2, 1, 3)[ma_input["x_key_valid_mask"]]
            else:
                sum_loss += cur_loss

        loss_dict['loss'] = sum_loss.item()
        for k, v in loss_dict.items():
            self.log(
                f'train/{k}',
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=len(data[-1]["scenario_id"])
            )
        return sum_loss

    def validation_step(self, data, batch_idx):
        memory_dict = None
        reg_loss_dict = {}
        all_outs = []

        self.eval()

        st = time.time()
        # iterate over prediction time steps
        for i in range(len(data)):
            # marginal predictions
            if i == len(data)-1: stl = time.time()
            cur_data = data[i]
            cur_data['memory_dict'] = memory_dict
            out = self(cur_data)
            _, cur_loss_dict = self.cal_loss(out, cur_data, tag=f'step{i}_')
            reg_loss_dict[f'val/step{i}_reg_loss'] = cur_loss_dict[f'step{i}_reg_loss']
            memory_dict = out['memory_dict']
            
            # joint predictions
            if self.ma:
                ma_input = self.getScenario(out, cur_data)
                scene_out = self.consistency_module(ma_input)
                cur_data["scored_mask"] = ma_input["x_key_valid_mask"]
                cur_data["target"] = ma_input["target"]

                y_hat = scene_out["y_hat"].permute(0, 2, 1, 3, 4)[ma_input["x_key_valid_mask"]]
                B = y_hat.shape[0]
                glo_y_hat = torch.bmm(y_hat.detach().reshape(B, -1, 2), torch.inverse(memory_dict["rot_mat"]))
                memory_dict["glo_y_hat"] = glo_y_hat.reshape(B, y_hat.size(1), -1, 2)
                memory_dict["x_mode"] = scene_out["x_mode"].permute(0, 2, 1, 3)[ma_input["x_key_valid_mask"]]
    
            all_outs.append(out)

        end = time.time()
        self.val_durations.append(end-st)
        self.last_val_durations.append(end-stl)

        # compute sa or ma latencies
        if self.ma:
            gt_len = ma_input['target'].shape[-2] 
            scene_out["y_hat"] = scene_out["y_hat"][:, :, :, :gt_len]
            metrics = self.metrics(scene_out, ma_input['target'], ma_input["x_key_valid_mask"])
        else:
            gt_len = data[-1]['target'][:, 0].shape[-2] 
            all_outs[-1]["y_hat"] = all_outs[-1]["y_hat"][:, :, :gt_len]
            metrics = self.metrics(all_outs[-1], data[-1]['target'][:, 0])

        self.log_dict(
            reg_loss_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        for k in self.val_scores.keys(): self.val_scores[k].append(metrics[k].item())
        return all_outs
    
    def test_step(self, data, batch_idx) -> None:
        memory_dict = None
        all_outs = []
        # todo bug fix
        for i in range(len(data)):
            cur_data = data[i]
            cur_data['memory_dict'] = memory_dict
            out = self(cur_data)
            memory_dict = out['memory_dict']
            all_outs.append(out)

        if self.ma:
            ma_input = self.getScenario(out, cur_data)
            scene_out = self.consistency_module(ma_input)
            self.submission_handler.format_data(ma_input, scene_out['y_hat'], scene_out['pi'])
        else:
            self.submission_handler.format_data(data[-1], all_outs[-1]['y_hat'][:, :, :60], all_outs[-1]['pi'])
    
    """
        Combine marginal predictions to joint scene input.
    """
    def getScenario(self, out, cur_data):
        if "bs_indices" not in cur_data.keys():
            ma_input = {
                'x_modes': out["memory_dict"]["x_mode"].unsqueeze(0),
                'origins': cur_data["origin"].unsqueeze(0),
                'thetas':  cur_data["theta"].unsqueeze(0),
                'x_key_valid_mask': torch.ones((1,  out["memory_dict"]["x_mode"].shape[0]), dtype=bool).cuda(),
                'y_hats': out["y_hat"],
                'pis': out["pi"]
            }
        else:
            from torch.nn.utils.rnn import pad_sequence
            bs_indices = torch.tensor(cur_data["bs_indices"])
            b = torch.unique(bs_indices).shape[0]

            x_modes = [out["memory_dict"]["x_mode"][bs_indices == i] for i in range(b)]
            x_modes = pad_sequence(x_modes, batch_first=True) 
            x_encoder = [out["memory_dict"]["x_encoder"][bs_indices == i] for i in range(b)]
            x_encoder = pad_sequence(x_encoder, batch_first=True) 
            origins = [cur_data["origin"][bs_indices == i] for i in range(b)]
            origins = pad_sequence(origins, batch_first=True) 
            thetas = [cur_data["theta"][bs_indices == i] for i in range(b)]
            thetas = pad_sequence(thetas, batch_first=True)    
            x_key_valid_mask = [cur_data["x_key_valid_mask"][:, 0][bs_indices == i] for i in range(b)]
            x_key_valid_mask = pad_sequence(x_key_valid_mask, batch_first=True,  padding_value=False)   

            y_hats = [out["y_hat"][bs_indices == i] for i in range(b)]
            y_hats = pad_sequence(y_hats, batch_first=True)   
            pi = [out["pi"][bs_indices == i] for i in range(b)]
            pi = pad_sequence(pi, batch_first=True)   

            if "target" in cur_data.keys():
                target = [cur_data["target"][:, 0][bs_indices == i] for i in range(b)]
                target = pad_sequence(target, batch_first=True)   
            else:
                target = torch.zeros_like(y_hats[:, :, 0])

            ma_input = {
                'x_modes': x_modes,
                'origins': origins,
                'thetas':  thetas,
                'x_key_valid_mask': x_key_valid_mask,
                'y_hats': y_hats,
                'pis': pi,
                'target': target,
                'x_encoder': x_encoder,
                'scenario_id': list(dict.fromkeys(cur_data["scenario_id"])),
                'track_id': [ np.array(cur_data["track_id"])[bs_indices.numpy() == i].tolist() for i in range(b) ]
            }
        return ma_input