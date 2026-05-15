<<<<<<< HEAD
# HcPCR++: Hyperbolic-Constraint Adaptive Curvature Point Cloud Reconstruction

This repository contains the implementation for:

**Hyperbolic-Constraint Adaptive Curvature Point Cloud Reconstruction from Single RGB-D Images**  
Wenrui Li, Zhe Yang, Hongtao Chen, Wangmeng Zuo, Xiaopeng Fan, and Yonghong Tian

HcPCR++ builds on the multiview compressive coding reconstruction pipeline and introduces hyperbolic constraints, adaptive curvature generation, and tensor-based geometry optimization to improve point cloud reconstruction from single RGB-D observations.

## Installation

Please see [INSTALL.md](INSTALL.md) for environment setup. The current code expects PyTorch, PyTorch3D, TensorLy, Geoopt, Open3D, and the other packages listed there.

## Data

Please see [DATASET.md](DATASET.md) for CO3D-v2 and Hypersim data preparation.

The CO3D metadata cache is loaded from `dataset_cache/`, and Hypersim preprocessing outputs such as `hypersim_gt_train.pt` and `hypersim_gt_val.pt` should be placed in the repository root as in the original NU-MCC pipeline.

## Checkpoints

 Put trained checkpoints under `pretrained/`, for example:

```bash
mkdir -p pretrained
# expected example path
pretrained/hcpcrpp.pth
```

You can pass any compatible checkpoint with `--resume [MODEL_PATH]` or `--checkpoint [MODEL_PATH]` for the demo scripts.

## CO3D-v2 Experiments

Train HcPCR++ from scratch:

```bash
PYTHONHASHSEED=[SEED] torchrun --nproc_per_node [NUM_GPU] main_hcpcrpp.py --exp_name [YOUR_EXPERIMENT_NAME] --accum_iter [32/NUM_GPU]
```

Example with 4 GPUs:

```bash
PYTHONHASHSEED=0 torchrun --nproc_per_node 4 main_hcpcrpp.py --exp_name hcpcrpp_udf --accum_iter 8
```

Evaluation and inference:

```bash
# Standard inference
PYTHONHASHSEED=[SEED] torchrun --nproc_per_node [N_GPU] main_hcpcrpp.py --run_val --resume [MODEL_PATH] --n_query_udf [BATCH_QUERY_FOR_REPULSIVE]

# High-resolution inference
PYTHONHASHSEED=[SEED] torchrun --nproc_per_node [N_GPU] main_hcpcrpp.py --run_val --resume [MODEL_PATH] --n_query_udf [BATCH_QUERY_FOR_REPULSIVE] --hr --xyz_size_hr 224

# Smoothing
PYTHONHASHSEED=[SEED] torchrun --nproc_per_node [N_GPU] main_hcpcrpp.py --run_val --resume [MODEL_PATH] --n_query_udf [BATCH_QUERY_FOR_REPULSIVE] --nneigh 12 --nn_seen 12
```

Example with 4 GPUs:

```bash
PYTHONHASHSEED=0 torchrun --nproc_per_node 4 main_hcpcrpp.py --run_val --resume pretrained/hcpcrpp.pth --n_query_udf 48000
```

Use `--run_viz` for visualization. Outputs are written to `experiments/[EXP_NAME]/viz/`. Use `--one_class [OBJECT_CLASS]` to evaluate or visualize one CO3D class, and `--save_pc` to export point clouds.

## Hypersim Experiments

Train on Hypersim:

```bash
torchrun --nproc_per_node 4 main_hcpcrpp.py --exp_name [EXPERIMENT_NAME] --hypersim_path [DATASET_PATH] --use_hypersim --blr 5e-5 --epochs 50 --train_epoch_len_multiplier 3200 --accum_iter 8 --n_groups 550
```

For evaluation or visualization, add `--run_val` or `--run_viz` and set `--resume [MODEL_PATH]`.

## Zero-shot Demos

Run reconstruction from an iPhone RGB-D capture:

```bash
python demo_iphone.py --checkpoint pretrained/hcpcrpp.pth
```

Run reconstruction from an RGB image plus depth and segmentation:

```bash
python demo_web.py --checkpoint pretrained/hcpcrpp.pth
```

The demo visualization is written to `demo/output.html`.

## Main Files

- `main_hcpcrpp.py`: training, evaluation, and visualization entry point.
- `src/model/hcpcrpp.py`: HcPCR++ model with adaptive curvature and tensor geometry modules.
- `src/engine/engine.py`: train/eval loops and reconstruction losses.
- `src/engine/engine_viz.py`: HTML and point cloud visualization utilities.
- `scripts/prepare_co3d.py` and `scripts/prepare_hypersim.py`: dataset preprocessing scripts.

## Acknowledgement

This codebase is adapted from [NU-MCC](https://github.com/sail-sg/numcc/tree/main), which is itself built on Meta's [MCC](https://github.com/facebookresearch/MCC). We thank the authors of both projects for releasing their code.

## Citation

If this code or paper is useful for your work, please cite:

```bibtex
@article{lihcpcrpp,
  title={Hyperbolic-Constraint Adaptive Curvature Point Cloud Reconstruction from Single RGB-D Images},
  author={Li, Wenrui and Yang, Zhe and Chen, Hongtao and Zuo, Wangmeng and Fan, Xiaopeng and Tian, Yonghong},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}
}
```
=======
# HcPCR
## Hyperbolic-Constraint Adaptive Curvature Point Cloud Reconstruction from Single RGB-D Images

>>>>>>> 48deb835533144e7547dbcec3d9531c44c740d29
