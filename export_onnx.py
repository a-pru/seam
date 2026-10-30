import inspect
import hydra
import numpy as np
import onnx
import torch

from src.deploy.dataset import build_val_loader
from src.deploy.runtime import OnnxSeam
from src.deploy.utils import load_model, onnx_paths, run_torch_sequence
from src.deploy.wrapper import INPUT_NAMES, MEMORY_KEYS, MEMORY_NAMES, OUTPUT_NAMES, SeamOnnxWrapper, batch_to_inputs


DIR = "./conf/"

def export(wrapper, inputs, input_names, path, cfg):
    # static shapes, legacy TorchScript-based exporter
    kwargs = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    torch.onnx.export(
        wrapper,
        tuple(inputs),
        str(path),
        input_names=input_names,
        output_names=OUTPUT_NAMES,
        opset_version=cfg.onnx.opset,
        do_constant_folding=True,
        **kwargs
    )
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    if cfg.onnx.simplify:
        from onnxsim import simplify
        model, ok = simplify(model)
        assert ok, f"onnxsim could not validate {path}"
        onnx.save(model, str(path))
    print(f"Exported {path}")


@hydra.main(version_base=None, config_path=DIR, config_name="config_onnx")
def main(cfg):
    # fast path of nn.MultiheadAttention has no ONNX export
    if hasattr(torch.backends, "mha"): torch.backends.mha.set_fastpath_enabled(False)

    model = load_model(cfg, device="cpu")
    steps = next(iter(build_val_loader(cfg)))
    assert steps[0]["x_centers"].shape[0] == cfg.batch_size

    # trace the streaming steps with real data: first step without memory, second step with memory
    with torch.no_grad():
        out = model({**steps[0], "memory_dict": None})
    memory = [out["memory_dict"][k] for k in MEMORY_KEYS]

    first_path, stream_path = onnx_paths(cfg)
    first_path.parent.mkdir(parents=True, exist_ok=True)
    # eval() on the wrapper, export restores its train/eval mode afterwards
    export(SeamOnnxWrapper(model, with_memory=False).eval(), batch_to_inputs(steps[0]), INPUT_NAMES, first_path, cfg)
    export(SeamOnnxWrapper(model, with_memory=True).eval(), batch_to_inputs(steps[1]) + memory, INPUT_NAMES + MEMORY_NAMES, stream_path, cfg)

    # compare ONNX Runtime with PyTorch over all streaming steps
    ref, _ = run_torch_sequence(model, steps)
    outs, _ = OnnxSeam(first_path, stream_path, providers=["CPUExecutionProvider"]).run_sequence(steps)
    max_diff = 0
    for i, (r, o) in enumerate(zip(ref, outs)):
        diff_y_hat, diff_pi = np.abs(r["y_hat"] - o["y_hat"]).max(), np.abs(r["pi"] - o["pi"]).max()
        print(f"step {i}: max abs diff y_hat {diff_y_hat:.2e}, pi {diff_pi:.2e}")
        max_diff = max(max_diff, diff_y_hat)
    assert max_diff < 1e-2, f"ONNX Runtime output deviates from PyTorch (max abs diff {max_diff:.2e})"


if __name__ == "__main__":
    main()
