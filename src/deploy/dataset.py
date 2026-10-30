import torch
from torch.utils.data import DataLoader

from src.datamodules.av2_dataset import Av2Dataset, collate_fn


AGENT_KEYS = [
    'target',
    'target_mask',
    'x_positions_diff',
    'x_positions',
    'x_attr',
    'x_centers',
    'x_angles',
    'x_velocity',
    'x_velocity_diff',
    'x_valid_mask',
]
LANE_KEYS = [
    'lane_positions',
    'lane_centers',
    'lane_angles',
    'lane_attr',
    'is_intersections',
    'lane_valid_mask',
]


def _keep_or_pad(item, keys, keep, size):
    for k in keys:
        if item.get(k) is None: continue
        v = item[k][keep]
        if v.shape[0] < size:
            v = torch.cat([v, v.new_zeros(size - v.shape[0], *v.shape[1:])], dim=0)
        item[k] = v


def fix_token_budget(item, num_agents, num_lanes):
    """
        Select the num_agents / num_lanes tokens closest to the focal agent and pad the rest (static shapes).
    """
    dist = torch.norm(item['x_centers'], dim=-1)
    dist[0] = -1 # focal agent stays at index 0
    keep = torch.topk(dist, min(num_agents, dist.shape[0]), largest=False).indices
    _keep_or_pad(item, AGENT_KEYS, keep, num_agents)

    dist = torch.norm(item['lane_centers'], dim=-1)
    keep = torch.topk(dist, min(num_lanes, dist.shape[0]), largest=False).indices
    _keep_or_pad(item, LANE_KEYS, keep, num_lanes)
    return item


# AV2 single-agent dataset with a fixed agent and lane token budget (MVA edge deployment)
class Av2DatasetFixed(Av2Dataset):
    def __init__(self, num_agents=20, num_lanes=155, **kwargs):
        super().__init__(**kwargs)
        assert not self.ma, "fixed token budget is only supported for single-agent data"
        self.num_agents = num_agents
        self.num_lanes = num_lanes

    def __getitem__(self, index: int):
        data = super().__getitem__(index)
        return [fix_token_budget(item, self.num_agents, self.num_lanes) for item in data]


def build_val_loader(cfg):
    dm_cfg = cfg.datamodule.pl_module
    dataset = Av2DatasetFixed(
        data_root=dm_cfg.data_root,
        split='val',
        num_agents=cfg.onnx.num_agents,
        num_lanes=cfg.onnx.num_lanes,
        **dm_cfg.dataset
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=dm_cfg.num_workers,
        pin_memory=dm_cfg.pin_memory,
        collate_fn=collate_fn,
    )
