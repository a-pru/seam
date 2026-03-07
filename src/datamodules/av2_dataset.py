from typing import List
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class Av2Dataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str = None,
        num_historical_steps: int = 50,
        split_points: List[int] = [50],
        radius: float = 150.0,
        logger=None,
        ma=False,
        num_future_steps=60
    ):
        assert split_points[-1] == 50 and num_historical_steps <= 50
        assert split in ['train', 'val', 'test', 'trainval']
        super(Av2Dataset, self).__init__()
        if split == 'trainval':
             self.file_list = sorted( list((Path(data_root)/"train").glob('*.pt')) + list((Path(data_root)/"val").glob('*.pt')) )
        else:
            self.data_folder = Path(data_root) / split
            self.file_list = sorted(list(self.data_folder.glob('*.pt')))
            if split == "train":
                self.file_list = self.file_list

        self.num_historical_steps = num_historical_steps
        self.num_future_steps = 0 if split =='test' else num_future_steps
        self.split_points = split_points
        self.radius = radius
        self.ma = ma
        print("AV2 DS SETTINGS:", self.num_historical_steps, self.num_future_steps, self.ma)
        
        if logger is not None:
            logger.info(f'data root: {data_root}/{split}, total number of files: {len(self.file_list)}')

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int):
        if self.ma: #all scored in extra batches
            raw_data = torch.load(self.file_list[index])
            indices = [raw_data["focal_idx"]] + raw_data["scored_idx"]
            data = [self.process(raw_data, idx) for idx in indices]
        else: #standard
            raw_data = torch.load(self.file_list[index])
            data = self.process(raw_data, raw_data["focal_idx"])
        return data
    
    def processMulti(self, data, indices):
        sequence_data = []
        for cur_step in self.split_points:
            ag_dict = self.process_multi_agent(data, indices, cur_step)
            sequence_data.append(ag_dict)
        return sequence_data        

    def process(self, data, idx):
        sequence_data = []
        for cur_step in self.split_points:
            ag_dict = self.process_single_agent(data, idx, cur_step)
            sequence_data.append(ag_dict)
        return sequence_data

    def process_multi_agent(self, data, indices, step=50):
        idx = indices[0]
        cur_agent_id = data['agent_ids'][idx]
        origin = data['x_positions'][idx, step - 1]
        theta = data['x_angles'][idx, step - 1]
        rotate_mat = torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta)],
                [torch.sin(theta), torch.cos(theta)],
            ],
        )
        ag_mask = torch.norm(data['x_positions'][:, step - 1] - origin, dim=-1) < self.radius
        ag_mask = ag_mask * data['x_valid_mask'][:, step - 1]
        for i in indices: ag_mask[i] = False

        # transform agents to local
        st, ed = step - self.num_historical_steps, step + self.num_future_steps
        attr = torch.cat([data['x_attr'][[i for i in indices]], data['x_attr'][ag_mask]])
        pos = data['x_positions'][:, st: ed]
        pos = torch.cat([pos[[i for i in indices]], pos[ag_mask]])
        head = data['x_angles'][:, st: ed]
        head = torch.cat([head[[i for i in indices]], head[ag_mask]])
        vel = data['x_velocity'][:, st: ed]
        vel = torch.cat([vel[[i for i in indices]], vel[ag_mask]])
        valid_mask = data['x_valid_mask'][:, st: ed]
        valid_mask = torch.cat([valid_mask[[i for i in indices]], valid_mask[ag_mask]])
  
        aids_int = []
        for a in data['agent_ids']:
            aids_int.append(int(a) if a != "AV" else 0)
        agent_ids = torch.tensor(aids_int)
        agent_ids = torch.cat( [agent_ids[i].unsqueeze(0) for i in indices] + [agent_ids[ag_mask]] )

        num_scored = len(indices)
        scored_mask = torch.zeros( (agent_ids.shape[0]), dtype=bool)
        scored_mask[:num_scored] = True

        pos[valid_mask] = torch.matmul(pos[valid_mask] - origin, rotate_mat)
        head[valid_mask] = (head[valid_mask] - theta + np.pi) % (2 * np.pi) - np.pi

        # transform lanes to local
        l_pos = data['lane_positions']
        l_attr = data['lane_attr']
        l_is_int = data['is_intersections']
        l_pos = torch.matmul(l_pos.reshape(-1, 2) - origin, rotate_mat).reshape(-1, l_pos.size(1), 2)

        l_ctr = l_pos[:, 9:11].mean(dim=1)
        l_head = torch.atan2(
            l_pos[:, 10, 1] - l_pos[:, 9, 1],
            l_pos[:, 10, 0] - l_pos[:, 9, 0],
        )
        l_valid_mask = (
            (l_pos[:, :, 0] > -self.radius) & (l_pos[:, :, 0] < self.radius)
            & (l_pos[:, :, 1] > -self.radius) & (l_pos[:, :, 1] < self.radius)
        )

        l_mask = l_valid_mask.any(dim=-1)
        l_pos = l_pos[l_mask]
        l_is_int = l_is_int[l_mask]
        l_attr = l_attr[l_mask]
        l_ctr = l_ctr[l_mask]
        l_head = l_head[l_mask]
        l_valid_mask = l_valid_mask[l_mask]

        l_pos = torch.where(
            l_valid_mask[..., None], l_pos, torch.zeros_like(l_pos)
        )

        # remove outliers
        nearest_dist = torch.cdist(pos[:, self.num_historical_steps - 1, :2],
                                   l_pos.view(-1, 2)).min(dim=1).values
        ag_mask = nearest_dist < 5
        ag_mask[:num_scored] = True
        pos = pos[ag_mask]
        head = head[ag_mask]
        vel = vel[ag_mask]
        attr = attr[ag_mask]
        valid_mask = valid_mask[ag_mask]
        scored_mask = scored_mask[ag_mask]

        # post_process
        head = head[:, :self.num_historical_steps]
        vel = vel[:, :self.num_historical_steps]
        pos_ctr = pos[:, self.num_historical_steps - 1].clone()
        if self.num_future_steps > 0:
            type_mask = attr[:, [-1]] != 3
            pos, target = pos[:, :self.num_historical_steps], pos[:, self.num_historical_steps:]
            target_mask = type_mask & valid_mask[:, [self.num_historical_steps - 1]] & valid_mask[:, self.num_historical_steps:]
            valid_mask = valid_mask[:, :self.num_historical_steps]
            target = torch.where(
                target_mask.unsqueeze(-1),
                target - pos_ctr.unsqueeze(1), torch.zeros_like(target),   
            )
        else:
            target = target_mask = None

        diff_mask = valid_mask[:, :self.num_historical_steps - 1] & valid_mask[:, 1: self.num_historical_steps]
        tmp_pos = pos.clone()
        pos_diff = pos[:, 1:self.num_historical_steps] - pos[:, :self.num_historical_steps - 1]
        pos[:, 1:self.num_historical_steps] = torch.where(
            diff_mask.unsqueeze(-1),
            pos_diff, torch.zeros(pos.size(0), self.num_historical_steps - 1, 2)
        )
        pos[:, 0] = torch.zeros(pos.size(0), 2)

        tmp_vel = vel.clone()
        vel_diff = vel[:, 1:self.num_historical_steps] - vel[:, :self.num_historical_steps - 1]
        vel[:, 1:self.num_historical_steps] = torch.where(
            diff_mask,
            vel_diff, torch.zeros(vel.size(0), self.num_historical_steps - 1)
        )
        vel[:, 0] = torch.zeros(vel.size(0))

        return {
            'target': target,
            'target_mask': target_mask,
            'x_positions_diff': pos,
            'x_positions': tmp_pos,
            'x_attr': attr,
            'x_centers': pos_ctr,
            'x_angles': head,
            'x_velocity': tmp_vel,
            'x_velocity_diff': vel,
            'x_valid_mask': valid_mask,
            'lane_positions': l_pos,
            'lane_centers': l_ctr,
            'lane_angles': l_head,
            'lane_attr': l_attr,
            'lane_valid_mask': l_valid_mask,
            'is_intersections': l_is_int,
            'origin': origin.view(1, 2),
            'theta': theta.view(1),
            'scenario_id': data['scenario_id'],
            'track_id': cur_agent_id,
            'city': data['city'],
            'timestamp': torch.Tensor([step * 0.1]),
            'agent_ids': agent_ids,
            'scored_mask': scored_mask
        }

    def process_single_agent(self, data, idx, step=50):
        cur_agent_id = data['agent_ids'][idx]
        origin = data['x_positions'][idx, step - 1]
        theta = data['x_angles'][idx, step - 1]
        rotate_mat = torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta)],
                [torch.sin(theta), torch.cos(theta)],
            ],
        )
        ag_mask = torch.norm(data['x_positions'][:, step - 1] - origin, dim=-1) < self.radius
        ag_mask = ag_mask * data['x_valid_mask'][:, step - 1]
        ag_mask[idx] = False

        # transform agents to local
        st, ed = step - self.num_historical_steps, step + self.num_future_steps
        attr = torch.cat([data['x_attr'][[idx]], data['x_attr'][ag_mask]])
        pos = data['x_positions'][:, st: ed]
        pos = torch.cat([pos[[idx]], pos[ag_mask]])
        head = data['x_angles'][:, st: ed]
        head = torch.cat([head[[idx]], head[ag_mask]])
        vel = data['x_velocity'][:, st: ed]
        vel = torch.cat([vel[[idx]], vel[ag_mask]])
        valid_mask = data['x_valid_mask'][:, st: ed]
        valid_mask = torch.cat([valid_mask[[idx]], valid_mask[ag_mask]])

        pos[valid_mask] = torch.matmul(pos[valid_mask] - origin, rotate_mat)
        head[valid_mask] = (head[valid_mask] - theta + np.pi) % (2 * np.pi) - np.pi

        # transform lanes to local
        l_pos = data['lane_positions']
        l_attr = data['lane_attr']
        l_is_int = data['is_intersections']
        l_pos = torch.matmul(l_pos.reshape(-1, 2) - origin, rotate_mat).reshape(-1, l_pos.size(1), 2)

        l_ctr = l_pos[:, 9:11].mean(dim=1)
        l_head = torch.atan2(
            l_pos[:, 10, 1] - l_pos[:, 9, 1],
            l_pos[:, 10, 0] - l_pos[:, 9, 0],
        )
        l_valid_mask = (
            (l_pos[:, :, 0] > -self.radius) & (l_pos[:, :, 0] < self.radius)
            & (l_pos[:, :, 1] > -self.radius) & (l_pos[:, :, 1] < self.radius)
        )

        l_mask = l_valid_mask.any(dim=-1)
        l_pos = l_pos[l_mask]
        l_is_int = l_is_int[l_mask]
        l_attr = l_attr[l_mask]
        l_ctr = l_ctr[l_mask]
        l_head = l_head[l_mask]
        l_valid_mask = l_valid_mask[l_mask]

        l_pos = torch.where(
            l_valid_mask[..., None], l_pos, torch.zeros_like(l_pos)
        )

        # remove outliers
        nearest_dist = torch.cdist(pos[:, self.num_historical_steps - 1, :2],
                                   l_pos.view(-1, 2)).min(dim=1).values
        ag_mask = nearest_dist < 5
        ag_mask[0] = True
        pos = pos[ag_mask]
        head = head[ag_mask]
        vel = vel[ag_mask]
        attr = attr[ag_mask]
        valid_mask = valid_mask[ag_mask]

        # post_process
        head = head[:, :self.num_historical_steps]
        vel = vel[:, :self.num_historical_steps]
        pos_ctr = pos[:, self.num_historical_steps - 1].clone()
        if self.num_future_steps > 0:
            type_mask = attr[:, [-1]] != 3
            pos, target = pos[:, :self.num_historical_steps], pos[:, self.num_historical_steps:]
            target_mask = type_mask & valid_mask[:, [self.num_historical_steps - 1]] & valid_mask[:, self.num_historical_steps:]
            valid_mask = valid_mask[:, :self.num_historical_steps]
            target = torch.where(
                target_mask.unsqueeze(-1),
                target - pos_ctr.unsqueeze(1), torch.zeros_like(target),   
            )
        else:
            target = target_mask = None

        diff_mask = valid_mask[:, :self.num_historical_steps - 1] & valid_mask[:, 1: self.num_historical_steps]
        tmp_pos = pos.clone()
        pos_diff = pos[:, 1:self.num_historical_steps] - pos[:, :self.num_historical_steps - 1]
        pos[:, 1:self.num_historical_steps] = torch.where(
            diff_mask.unsqueeze(-1),
            pos_diff, torch.zeros(pos.size(0), self.num_historical_steps - 1, 2)
        )
        pos[:, 0] = torch.zeros(pos.size(0), 2)

        tmp_vel = vel.clone()
        vel_diff = vel[:, 1:self.num_historical_steps] - vel[:, :self.num_historical_steps - 1]
        vel[:, 1:self.num_historical_steps] = torch.where(
            diff_mask,
            vel_diff, torch.zeros(vel.size(0), self.num_historical_steps - 1)
        )
        vel[:, 0] = torch.zeros(vel.size(0))

        return {
            'target': target,
            'target_mask': target_mask,
            'x_positions_diff': pos,
            'x_positions': tmp_pos,
            'x_attr': attr,
            'x_centers': pos_ctr,
            'x_angles': head,
            'x_velocity': tmp_vel,
            'x_velocity_diff': vel,
            'x_valid_mask': valid_mask,
            'lane_positions': l_pos,
            'lane_centers': l_ctr,
            'lane_angles': l_head,
            'lane_attr': l_attr,
            'lane_valid_mask': l_valid_mask,
            'is_intersections': l_is_int,
            'origin': origin.view(1, 2),
            'theta': theta.view(1),
            'scenario_id': data['scenario_id'],
            'track_id': cur_agent_id,
            'city': data['city'],
            'timestamp': torch.Tensor([step * 0.1])
        }


def collate_fn(seq_batch):
    seq_data = []
    ts = len(seq_batch[0][0]) if isinstance(seq_batch[0][0], list) else len(seq_batch[0])
    for i in range(ts):
        data = {}
        if isinstance(seq_batch[0][0], list):
            tmp = [item for sublist in seq_batch for item in sublist]
            batch = [b[i] for b in tmp]
            data["bs_indices"] = [i for i, sublist in enumerate(seq_batch) for _ in sublist]
        else:
            batch = [b[i] for b in seq_batch]

        for key in [
            'x_positions_diff',
            'x_attr',
            'x_positions',
            'x_centers',
            'x_angles',
            'x_velocity',
            'x_velocity_diff',
            'lane_positions',
            'lane_centers',
            'lane_angles',
            'lane_attr',
            'is_intersections',
        ]:
            data[key] = pad_sequence([b[key] for b in batch], batch_first=True)

        if 'agent_ids' in batch[0]:
            data['agent_ids'] = pad_sequence(
                [b['agent_ids'] for b in batch], batch_first=True
            )
            data['scored_mask'] = pad_sequence(
                [b['scored_mask'] for b in batch], batch_first=True, padding_value=False
            )

        if batch[0]['target'] is not None:
            data['target'] = pad_sequence([b['target'] for b in batch], batch_first=True)
            data['target_mask'] = pad_sequence(
                [b['target_mask'] for b in batch], batch_first=True, padding_value=False
            )

        for key in ['x_valid_mask', 'lane_valid_mask']:
            data[key] = pad_sequence(
                [b[key] for b in batch], batch_first=True, padding_value=False
            )

        data['x_key_valid_mask'] = data['x_valid_mask'].any(-1)
        data['lane_key_valid_mask'] = data['lane_valid_mask'].any(-1)

        data['scenario_id'] = [b['scenario_id'] for b in batch]
        data['track_id'] = [b['track_id'] for b in batch]

        data['origin'] = torch.cat([b['origin'] for b in batch], dim=0)
        data['theta'] = torch.cat([b['theta'] for b in batch])
        data['timestamp'] = torch.cat([b['timestamp'] for b in batch])
        seq_data.append(data)
    return seq_data
