import time

import numpy as np
import onnxruntime as ort

from .wrapper import INPUT_NAMES, MEMORY_NAMES, NEW_MEMORY_NAMES, batch_to_inputs


def _pad_batch(x, batch_size):
    # repeat the last sample to fill up a short batch
    if x.shape[0] == batch_size:
        return x
    return np.concatenate([x, np.repeat(x[-1:], batch_size - x.shape[0], axis=0)], axis=0)


# Streaming SEAM inference with ONNX Runtime
# first_path: graph for the first streaming step, stream_path: graph for all later steps (takes the memory as input)
class OnnxSeam:
    def __init__(self, first_path, stream_path, providers=None):
        providers = providers or ort.get_available_providers()
        self.first = ort.InferenceSession(str(first_path), providers=providers)
        self.stream = ort.InferenceSession(str(stream_path), providers=providers)
        self.batch_size = self.first.get_inputs()[0].shape[0]

    def step(self, inputs, memory=None):
        """
            Run one streaming step. inputs / memory: dicts of numpy arrays with the exported batch size.
        """
        session = self.first if memory is None else self.stream
        feed = dict(inputs, **(memory or {}))
        feed = {i.name: feed[i.name] for i in session.get_inputs()}
        out = dict(zip([o.name for o in session.get_outputs()], session.run(None, feed)))
        memory = {name: out[new_name] for name, new_name in zip(MEMORY_NAMES, NEW_MEMORY_NAMES)}
        return out, memory

    def run_sequence(self, steps):
        """
            Run all streaming steps of a collated batch (list of dicts, one per split point).
            Returns per step outputs (y_hat, pi) as numpy arrays and per step latencies in seconds.
        """
        B = steps[0]['x_centers'].shape[0]
        assert B <= self.batch_size, f"batch size {B} exceeds exported batch size {self.batch_size}"
        memory, outs, durations = None, [], []
        for cur_data in steps:
            inputs = {n: _pad_batch(t.cpu().numpy(), self.batch_size) for n, t in zip(INPUT_NAMES, batch_to_inputs(cur_data))}
            st = time.time()
            out, memory = self.step(inputs, memory)
            durations.append(time.time() - st)
            outs.append({'y_hat': out['y_hat'][:B], 'pi': out['pi'][:B]})
        return outs, durations
