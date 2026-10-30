import hydra
import numpy as np
import torch
from torchmetrics import MetricCollection
from tqdm import tqdm

from src.deploy.dataset import build_val_loader
from src.deploy.utils import load_model, onnx_paths, run_torch_sequence
from src.metrics import MR, minADE, minFDE, brier_minFDE


DIR = "./conf/"

@hydra.main(version_base=None, config_path=DIR, config_name="config_onnx")
def main(cfg):
    if cfg.onnx.backend == "ort":
        from src.deploy.runtime import OnnxSeam
        first_path, stream_path = onnx_paths(cfg)
        runner = OnnxSeam(first_path, stream_path, providers=list(cfg.onnx.providers))
        print("ONNX Runtime providers:", runner.first.get_providers())
        run_sequence = runner.run_sequence
    else:
        model = load_model(cfg, device="cuda" if torch.cuda.is_available() else "cpu")
        run_sequence = lambda steps: run_torch_sequence(model, steps)

    metrics = MetricCollection(
        {
            'minADE1': minADE(k=1),
            'minADE6': minADE(k=6),
            'minFDE1': minFDE(k=1),
            'minFDE6': minFDE(k=6),
            'MR': MR(),
            'b-minFDE6': brier_minFDE(k=6),
        }
    )
    scores = {k: [] for k in ["MR", "minADE1", "minADE6", "minFDE1", "minFDE6", "b-minFDE6"]}
    durations = []

    for steps in tqdm(build_val_loader(cfg)):
        outs, step_durations = run_sequence(steps)
        durations.append(step_durations)

        # evaluate prediction of the last streaming step
        target = steps[-1]["target"][:, 0]
        out = {
            "y_hat": torch.from_numpy(outs[-1]["y_hat"])[:, :, :target.shape[-2]],
            "pi": torch.from_numpy(outs[-1]["pi"]),
        }
        batch_scores = metrics(out, target)
        for k in scores.keys(): scores[k].append(batch_scores[k].item())

    # average over batches as in eval.py
    print(f"Backend: {cfg.onnx.backend}, token budget: {cfg.onnx.num_agents} agents / {cfg.onnx.num_lanes} lanes")
    print("| " + " | ".join(scores.keys()) + " |")
    print("| " + " | ".join(":---" for _ in scores) + " |")
    print("| " + " | ".join("{:.3f}".format(np.mean(v)) for v in scores.values()) + " |")
    # skip first batch (warm-up)
    lat = np.mean(durations[1:], axis=0) * 1000
    print("Latency per streaming step [ms]: " + ", ".join("{:.1f}".format(l) for l in lat))


if __name__ == "__main__":
    main()
