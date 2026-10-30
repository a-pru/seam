import torch
import torch.nn as nn


# model inputs per streaming step (batch dimension first)
INPUT_NAMES = [
    'x_positions_diff',
    'x_attr',
    'x_centers',
    'x_angles',
    'x_velocity_diff',
    'x_valid_mask',
    'x_key_valid_mask',
    'lane_positions',
    'lane_centers',
    'lane_angles',
    'lane_valid_mask',
    'lane_key_valid_mask',
    'origin',
    'theta',
    'timestamp',
]
# streaming memory passed from one step to the next
MEMORY_KEYS = ['x_encoder', 'x_mode', 'glo_y_hat', 'x_mask', 'origin', 'theta', 'timestamp']
MEMORY_NAMES = ['memory_' + k for k in MEMORY_KEYS]
NEW_MEMORY_NAMES = ['new_' + n for n in MEMORY_NAMES]
OUTPUT_NAMES = ['y_hat', 'pi'] + NEW_MEMORY_NAMES


def batch_to_inputs(cur_data):
    """
        Collated batch of one streaming step -> list of input tensors ordered as INPUT_NAMES.
    """
    inputs = [cur_data[name] for name in INPUT_NAMES]
    inputs[INPUT_NAMES.index('x_attr')] = cur_data['x_attr'].long()
    return inputs


# Flat tensor interface around SeamOnnx for torch.onnx.export
# with_memory=False: first streaming step, with_memory=True: all later steps
class SeamOnnxWrapper(nn.Module):
    def __init__(self, model, with_memory):
        super().__init__()
        self.model = model
        self.with_memory = with_memory

    def forward(self, *inputs):
        data = dict(zip(INPUT_NAMES, inputs[:len(INPUT_NAMES)]))
        data['memory_dict'] = None
        if self.with_memory:
            memory_dict = dict(zip(MEMORY_KEYS, inputs[len(INPUT_NAMES):]))
            # constant for a fixed token budget, agent tokens come first
            B, N_a, N_l = data['x_centers'].shape[0], data['x_centers'].shape[1], data['lane_centers'].shape[1]
            memory_dict['x_type_mask'] = torch.cat([
                torch.ones(B, N_a, dtype=torch.bool, device=data['x_centers'].device),
                torch.zeros(B, N_l, dtype=torch.bool, device=data['x_centers'].device)
            ], dim=1)
            data['memory_dict'] = memory_dict

        out = self.model(data)
        return tuple([out['y_hat'], out['pi']] + [out['memory_dict'][k] for k in MEMORY_KEYS])
