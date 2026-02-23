# Streaming Real-Time Trajectory Prediction Using Endpoint-Aware Modeling (SEAM)
**(Code and model weights coming soon!)**

### Paper
> **Streaming Real-Time Trajectory Prediction Using Endpoint-Aware Modeling**           
> Alexander Prutsch, David Schinagl, Horst Possegger
> **Graz University of Technology**  
> **WACV 2026 Oral**

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

## Training on Single-Agent benchmark
Train SEAM model using
```
python train.py datamodule.pl_module.data_root=/path/to/data_root/seam_processed/
```

## Evaluation on Single-Agent benchmark
Evaluate SEAM model using
```
python train.py datamodule.pl_module.data_root=/path/to/data_root/seam_processed/ checkpoint=outputs/path/to/experiment/checkpoint_file.ckpt
```

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
 author={Alexander Prutsch, David Schinagl, Horst Possegger},
 booktitle={WACV},
 year={2026},
}
```

## Acknowledgements
This repository is based on [RealMotion](https://github.com/fudan-zvg/RealMotion/) and integrates code from [Forecast-MAE](https://github.com/jchengai/forecast-mae) and [EMP](https://github.com/a-pru/emp). We thank them for their work!
