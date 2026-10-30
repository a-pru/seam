# Streaming Real-Time Trajectory Prediction Using Endpoint-Aware Modeling (SEAM)

### [[Paper]](https://openaccess.thecvf.com/content/WACV2026/html/Prutsch_Streaming_Real-Time_Trajectory_Prediction_Using_Endpoint-Aware_Modeling_WACV_2026_paper.html) [[ArXiv]](https://arxiv.org/abs/2603.01864) [[MVA Journal]](https://link.springer.com/article/10.1007/s00138-026-01939-1)

> [!NOTE]
> **Check out [SHARP](https://github.com/a-pru/sharp) (CVPR 2026), the evolved version of SEAM**, with improved accuracy, robustness, and a broad code release for nuScenes, Argoverse 1, and Argoverse 2, including the multi-agent implementation.

> [**Streaming Real-Time Trajectory Prediction Using Endpoint-Aware Modeling**](https://arxiv.org/abs/2603.01864)  
> Alexander Prutsch, David Schinagl, Horst Possegger  
> **Graz University of Technology**  
> **WACV 2026 Oral**

> [**Edge-Deployed Streaming Real-Time Trajectory Prediction**](https://link.springer.com/article/10.1007/s00138-026-01939-1)  
> Alexander Prutsch, Erik Bonek, David Schinagl, Matthias Wess, Horst Possegger  
> **Graz University of Technology, TU Wien**  
> **Machine Vision and Applications (MVA) 2026**

Our MVA journal extension deploys SEAM on an embedded NVIDIA Jetson Orin. This repository includes the corresponding [ONNX export and inference](#onnx-export-and-inference-mva) code.

## Getting Started

### Create and Activate Virtual Environment
```
conda create -n seam python=3.11.10
conda activate seam
```

### Install PyTorch
We tested our implementation with torch 2.1.1 and CUDA 12.1.

Install PyTorch e.g.
```
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Install Dependencies
```
pip install -r ./requirements.txt
```

### Download Download [Argoverse 2 Motion Forecasting Dataset](https://argoverse.github.io/user-guide/datasets/motion_forecasting.html#download)
The expected structure of the AV2 data should be:
```
data_root
    ├── train
    │   ├── 0000b0f9-99f9-4a1f-a231-5be9e4c523f7
    │   ├── 0000b6ab-e100-4f6b-aee8-b520b57c0530
    │   ├── ...
    ├── val
    │   ├── 00010486-9a07-48ae-b493-cf4545855937
    │   ├── 00062a32-8d6d-4449-9948-6fedac67bfcd
    │   ├── ...
    ├── test
    │   ├── 0000b329-f890-4c2b-93f2-7e2413d4ca5b
    │   ├── 0008c251-e9b0-4708-b762-b15cb6effc27
    │   ├── ...
```

### Data Preprocessing
Preprocess the Argoverse 2 dataset by executing
```
python preprocess.py --data_root=/path/to/data_root -p
```

## Training on Single-Agent Benchmark
Train SEAM model using
```
python train.py datamodule.pl_module.data_root=/path/to/data_root/seam_processed/
```

## Evaluation on Single-Agent Benchmark
**Checkpoint provided in the repository** (stored with [Git LFS](https://git-lfs.com): run `git lfs install` before cloning or `git lfs pull` after cloning).

Evaluate SEAM model using
```
python eval.py datamodule.pl_module.data_root=/path/to/data_root/seam_processed/ checkpoint=checkpoints/seam.ckpt
```

Expected results:
| MR | minADE1 | minADE6 | minFDE1 | minFDE6 | b-minFDE6 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.153 | 1.598 | 0.663 | 3.962 | 1.249 | 1.848 |

## Training and Evaluation on Multi-Agent Benchmark
**Code included, checkpoint and instructions coming soon.**

## ONNX Export and Inference (MVA)
Following our [MVA journal extension](https://link.springer.com/article/10.1007/s00138-026-01939-1), SEAM can be exported to ONNX with static input shapes for deployment on edge hardware (e.g., TensorRT on an NVIDIA Jetson).
Static shapes require a fixed batch size and a fixed token budget: per focal agent, the `num_agents` closest agents and `num_lanes` closest lane segments are kept (padded if fewer).
The static-shape model is implemented in [`src/deploy/seam_onnx.py`](src/deploy/seam_onnx.py) and uses the same weights as SEAM.

### Install Dependencies
```
pip install -r ./requirements_onnx.txt
```
We tested the ONNX export with torch 2.8.0 and onnxruntime 1.31.
The CUDA execution provider requires an onnxruntime-gpu build that matches your CUDA and cuDNN versions (see the [requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)), otherwise ONNX Runtime falls back to the CPU.

### Inference with ONNX Runtime
**ONNX models (batch size 32, 20 agents, 155 lanes) provided in the repository** (`checkpoints/onnx/`).

SEAM is a streaming model: `*_first.onnx` processes the first frame and `*_stream.onnx` all subsequent frames.
The stream model additionally takes the streaming memory (`memory_*` inputs) returned by the previous step (`new_memory_*` outputs).
See [`src/deploy/runtime.py`](src/deploy/runtime.py) for the streaming inference loop.

Evaluate the ONNX models on the AV2 validation set using
```
python eval_onnx.py datamodule.pl_module.data_root=/path/to/data_root/seam_processed/
```
Set `onnx.backend=torch` to evaluate the PyTorch model with the same token budget.

Expected results:
| Model | Agents / Lanes | MR | minADE1 | minADE6 | minFDE1 | minFDE6 | b-minFDE6 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ONNX Runtime | 20 / 155 | 0.158 | 1.632 | 0.674 | 4.077 | 1.286 | 1.887 |
| PyTorch | 20 / 155 | 0.157 | 1.632 | 0.674 | 4.077 | 1.286 | 1.887 |
| PyTorch | all | 0.153 | 1.598 | 0.663 | 3.962 | 1.249 | 1.848 |

### Export to ONNX
Export SEAM to ONNX (opset 17, simplified with [onnxsim](https://github.com/onnxsim/onnxsim)) using
```
python export_onnx.py datamodule.pl_module.data_root=/path/to/data_root/seam_processed/ checkpoint=checkpoints/seam.ckpt onnx.num_agents=20 onnx.num_lanes=155 batch_size=32
```
The script traces the model on validation data and checks the exported models against PyTorch with ONNX Runtime.
Batch size and token budget are fixed in the exported models.

### TensorRT
Build TensorRT engines (e.g., on a Jetson) with `trtexec`:
```
trtexec --onnx=checkpoints/onnx/seam_B32_A20_L155_first.onnx --saveEngine=seam_first.engine --fp16
trtexec --onnx=checkpoints/onnx/seam_B32_A20_L155_stream.onnx --saveEngine=seam_stream.engine --fp16
```
Alternatively, run the ONNX models with the TensorRT execution provider of ONNX Runtime (`onnx.providers=[TensorrtExecutionProvider,CUDAExecutionProvider]`).
In our MVA paper, this configuration (batch size 32, 20 agents, 155 lanes) runs in 97 ms on an NVIDIA Jetson Orin Nano with FP16 precision.

## Visualize Results
Visualize the prediction results using
```
python visualize.py
```

Please update the data_root, chkpt_dir, and av2_raw_data_dir variable in the script.

## Bibtex
```bibtex
@inproceedings{prutsch2026streaming,
    title={{Streaming Real-Time Trajectory Prediction Using Endpoint-Aware Modeling}},
    author={Prutsch, Alexander and Schinagl, David and Possegger, Horst},
    booktitle={In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
    year={2026}
}

@article{prutsch2026edge,
    title={{Edge-Deployed Streaming Real-Time Trajectory Prediction}},
    author={Prutsch, Alexander and Bonek, Erik and Schinagl, David and Wess, Matthias and Possegger, Horst},
    journal={Machine Vision and Applications},
    volume={38},
    number={1},
    pages={7},
    year={2026},
    doi={10.1007/s00138-026-01939-1}
}
```

## Acknowledgements
This repository is based on [RealMotion](https://github.com/fudan-zvg/RealMotion/) and integrates code from [Forecast-MAE](https://github.com/jchengai/forecast-mae) and [EMP](https://github.com/a-pru/emp). We thank them for their work!
