# LiDAR 3D Object Detection — TensorRT-Optimized PointPillars

🚧 **Work in progress.** This repo is being built in public as a portfolio project — commits reflect real, incremental progress rather than a polished final drop. See [Roadmap](#roadmap) for what's left.

## Overview

PointPillars LiDAR 3D object detection ([Lang et al., 2019](https://arxiv.org/abs/1812.05784)), being deployed and optimized with TensorRT for real-time inference on KITTI. The goal isn't just "export a model" — MMDeploy's own documentation confirms that PointPillars' voxelization (pre-processing) and NMS/box-decoding (post-processing) steps **cannot** be exported into the ONNX graph. This project implements both as standalone components wrapped around the TensorRT engine, which is the actual differentiator over a generic export tutorial.

Built as a flagship portfolio piece targeting 3D computer vision / LiDAR perception / robotics engineering roles.

## Status

| # | Milestone | Status |
|---|---|---|
| 1 | PyTorch baseline (official pretrained KITTI-3class checkpoint) | ✅ Done — validated |
| 2 | ONNX export via MMDeploy | ✅ Done — validated |
| 3 | TensorRT engine, FP32 → FP16 | 🔲 Not started |
| 4 | Custom voxelize + NMS/decode wrapper | 🔲 Not started — the core technical challenge |
| 5 | Benchmark: PyTorch vs ONNX Runtime vs TensorRT (latency, throughput, mAP) | 🔲 Not started |
| 6 | INT8 quantization | ⬜ Stretch goal |
| 7 | ROS 2 wrapper | ⬜ Stretch goal |
| 8 | Jetson compatibility analysis (written, not hardware-validated — no device available) | ⬜ Stretch goal |

## Baseline results

Official pretrained checkpoint (`hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class`), evaluated on the standard KITTI val split (3,769 frames):

| Metric | This run | Published (MMDetection3D model zoo) | Diff |
|---|---|---|---|
| AP40, Overall, Moderate, 3D | **64.36** | 64.07 | ~0.3 (normal run-to-run variance) |

Validated three independent ways, not just the aggregate number:
1. **Aggregate mAP** matches the published baseline (above).
2. **Visual sanity check** — `demo/pcd_demo.py --show` on real val-split frames; predicted boxes are spatially and dimensionally sane.
3. **Hand-checked single-object match** — top-scoring prediction on frame `000042` compared directly against its ground-truth label (camera→LiDAR frame conversion done by hand): matched within ~0.4m position and centimeters on dimensions.

Full log: [`results/baseline_eval_log.txt`](results/baseline_eval_log.txt).

### ONNX export validation

Exported via MMDeploy and re-evaluated on the full val split, to confirm the export preserved model correctness before touching TensorRT:

| Metric | PyTorch (baseline) | ONNX Runtime (exported) | Diff |
|---|---|---|---|
| AP40, Overall, Moderate, 3D | 64.3590 | **64.3536** | 0.0054 |

Effectively a rounding-level difference — confirms the exported graph (backbone + detection head) is numerically faithful to the PyTorch model. This does **not** yet validate the standalone voxelize/NMS wrapper, which still relies on MMDeploy's own Python-side pre/post-processing at eval time — that remains the project's core unsolved milestone. Full writeup, including the environment issues hit along the way: [`docs/onnx-export.md`](docs/onnx-export.md).

These two rows are the "before" data for the eventual PyTorch vs. ONNX Runtime vs. TensorRT FP32/FP16 benchmark table.

## Key technical decisions

| Decision | Choice | Why |
|---|---|---|
| Model architecture | PointPillars (MMDetection3D) | Voxel/pillar-based — avoids the sparse-conv custom-CUDA export wall that makes other 3D architectures far riskier to deploy under TensorRT |
| Training approach | Official pretrained checkpoint, not trained from scratch | Project's value is the deployment/optimization work, not re-proving training competency |
| Environment isolation | `venv`, not conda | Matches the deployment target's own convention — conda is known to conflict with ROS2's `colcon`/`ament` build system, and Jetson/JetPack itself doesn't use conda |
| CUDA version | 11.8 | Matched to the TensorRT version below |
| **TensorRT version** | **10.12, not the newer 11.x line** | **TensorRT 11.2.1 has no Jetson/JetPack support at all** ([NVIDIA docs](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html)) — Jetson Orin deployments must stay on a TensorRT 10.x release. Building against 11.x would make any Jetson-feasibility analysis reference a TensorRT line that can't actually run on the target hardware. |
| TensorRT install method | tar, not deb | No root required, fully self-contained/portable, multiple versions can coexist — consistent with the `venv` choice above |

## The core challenge: voxelize/NMS aren't exportable

PointPillars needs two steps MMDeploy [documents as unsupported for ONNX export](https://github.com/open-mmlab/mmdetection3d/issues/3135):

- **Voxelization** (pre-processing) — bins raw LiDAR points into pillar-based features. Involves dynamic-size, scatter/gather operations that don't map onto a static ONNX graph.
- **NMS / box decoding** (post-processing) — turns raw anchor-offset predictions into final 3D boxes and removes duplicates via Non-Maximum Suppression. Inherently variable-length, another poor fit for static graph export.

**Plan:** implement both as standalone Python/PyTorch modules wrapping the TensorRT engine — reusing MMDetection3D's own existing `Voxelization` and `nms_rotated` logic rather than reimplementing the math from scratch, verified against the full PyTorch pipeline's output before wiring them around the exported engine. Reference: [NVIDIA-AI-IOT/CUDA-PointPillars](https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars) documents the same three-phase pattern.

This isn't yet implemented — tracked as the current top-priority milestone.

## Repository structure

```
lidar-pointpillars-tensorrt/
├── README.md
├── requirements.txt
├── docs/
│   └── onnx-export.md
└── results/
    ├── baseline_eval_log.txt
    └── onnx_export_eval_log.txt
```

Full target structure (populated incrementally, not scaffolded in advance):

```
├── configs/           # train + deploy configs
├── src/
│   ├── preprocess/    # voxelization (the ONNX gap, solved)
│   ├── postprocess/   # NMS/box decode (the other ONNX gap, solved)
│   ├── export/        # ONNX + TensorRT engine build
│   └── inference/     # end-to-end inference wrapper
├── benchmark/         # latency/throughput/accuracy comparison
├── demo/              # single-sample inference + visualization
└── docs/              # architecture + full benchmark methodology
```

## Setup (current — baseline + ONNX export)

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires: KITTI dataset (Velodyne, calib, labels, **and** `image_2` — needed even for LiDAR-only training, since KITTI's camera-FOV frustum culling requires image dimensions) registered/downloaded separately from [cvlibs.net/datasets/kitti](https://www.cvlibs.net/datasets/kitti), processed via MMDetection3D's `create_data.py`.

**Note:** `requirements.txt` deliberately excludes TensorRT, MMDetection3D, and MMDeploy — TensorRT is installed manually from NVIDIA's tar distribution (not PyPI), and MMDetection3D/MMDeploy are editable installs from cloned repos, not published pip packages. `pip freeze` would otherwise emit broken or misleading lines for all three. See the [Environment table](#environment-verified-working) below and [`docs/onnx-export.md`](docs/onnx-export.md) for the manual install steps and version-pinning fixes required for each.

### Environment (verified working)

| Component | Version |
|---|---|
| OS | Ubuntu 22.04 |
| GPU | RTX 3060, 12GB |
| NVIDIA Driver | 535.183.01 |
| CUDA Toolkit | 11.8 |
| cuDNN | 8.9.7 (built for CUDA 11.x) |
| TensorRT | 10.12 (tar install) |
| MMDetection3D | 1.4.0 (commit `962f0937`) |
| MMDeploy | 1.3.1 |
| onnxruntime-gpu | 1.16.3 |
| Python | 3.10 |

Pip-installable dependencies are pinned in [`requirements.txt`](requirements.txt) — the table above covers system-level and manually-installed dependencies `pip freeze` can't reliably capture (see note above). If you hit a version-mismatch error, check this table before anything else — it's the single most common failure mode across this whole ecosystem.

TensorRT engine-build setup instructions will be added once that stage lands.

## Roadmap

- [x] ONNX export via MMDeploy — validated against ONNX Runtime, AP40 moderate 3D within 0.0054 of PyTorch baseline. See [`docs/onnx-export.md`](docs/onnx-export.md).
- [ ] Standalone voxelize + NMS/decode modules, verified against PyTorch baseline output
- [ ] TensorRT engine build (FP32 → FP16) via TensorRT 10.12 Python API
- [ ] Full benchmark: latency, throughput, and mAP across PyTorch / ONNX Runtime / TensorRT
- [ ] Written Jetson Orin feasibility analysis (compute/memory/power envelope, based on published specs — not hardware-validated, no device available)
- [ ] Stretch: INT8 quantization, ROS 2 wrapper

## Acknowledgements

Built on [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) and [MMDeploy](https://github.com/open-mmlab/mmdeploy). PointPillars: Lang et al., 2019.

## License

MIT — see [LICENSE](LICENSE).