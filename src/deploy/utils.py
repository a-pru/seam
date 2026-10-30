import time
from pathlib import Path

import torch
from hydra.utils import instantiate


def load_model(cfg, device):
    pl_module = instantiate(cfg.model.pl_module)
    ckpt = torch.load(cfg.checkpoint, map_location='cpu', weights_only=False)['state_dict']
    pl_module.load_state_dict(ckpt)
    return pl_module.model.eval().to(device)


def onnx_paths(cfg):
    # batch size and token budget are baked into the exported graphs
    name = f"seam_B{cfg.batch_size}_A{cfg.onnx.num_agents}_L{cfg.onnx.num_lanes}"
    return [Path(cfg.onnx.out_dir) / f"{name}_{step}.onnx" for step in ("first", "stream")]


@torch.no_grad()
def run_torch_sequence(model, steps):
    """
        PyTorch counterpart of OnnxSeam.run_sequence.
    """
    device = next(model.parameters()).device
    memory, outs, durations = None, [], []
    for cur_data in steps:
        cur_data = {k: v.to(device) if torch.is_tensor(v) else v for k, v in cur_data.items()}
        cur_data['memory_dict'] = memory
        if device.type == 'cuda': torch.cuda.synchronize()
        st = time.time()
        out = model(cur_data)
        if device.type == 'cuda': torch.cuda.synchronize()
        durations.append(time.time() - st)
        memory = out['memory_dict']
        outs.append({'y_hat': out['y_hat'].cpu().numpy(), 'pi': out['pi'].cpu().numpy()})
    return outs, durations
