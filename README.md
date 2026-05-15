# HcPCR++: Hyperbolic-Constraint Adaptive Curvature Point Cloud Reconstruction
Wenrui Li, Zhe Yang, Hongtao Chen, Wangmeng Zuo, Xiaopeng Fan, and Yonghong Tian

## Installation

Please see [INSTALL.md](INSTALL.md) for environment setup. The current code expects PyTorch, PyTorch3D, TensorLy, Geoopt, Open3D, and the other packages listed there.

## Data

Please see [DATASET.md](DATASET.md) for CO3D-v2 and Hypersim data preparation.

The CO3D metadata cache is loaded from `dataset_cache/`, and Hypersim preprocessing outputs such as `hypersim_gt_train.pt` and `hypersim_gt_val.pt` should be placed in the repository root as in the original NU-MCC pipeline.

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
```

Example with 4 GPUs:

```bash
PYTHONHASHSEED=0 torchrun --nproc_per_node 4 main_hcpcrpp.py --run_val --resume pretrained/hcpcrpp.pth --n_query_udf 48000
```

Use `--run_viz` for visualization. Outputs are written to `experiments/[EXP_NAME]/viz/`. Use `--one_class [OBJECT_CLASS]` to evaluate or visualize one CO3D class, and `--save_pc` to export point clouds.

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
## Acknowledgement

This codebase is adapted from [NU-MCC](https://github.com/sail-sg/numcc/tree/main), which is itself built on Meta's [MCC](https://github.com/facebookresearch/MCC). We thank the authors of both projects for releasing their code.
