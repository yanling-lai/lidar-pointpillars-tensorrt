# ONNX Export (via MMDeploy)

Exports the PyTorch baseline checkpoint to ONNX and validates it against ONNX Runtime, isolating export correctness from TensorRT correctness before touching TensorRT at all.

**Known limitation (by design, not a bug):** MMDeploy [documents that voxelization (pre-processing) and NMS/box-decoding (post-processing) are not converted into ONNX ops for mmdet3d models](https://github.com/open-mmlab/mmdetection3d/issues/3135) — this export covers the backbone + detection head only. The evaluation below still runs correctly because MMDeploy's own test pipeline handles voxelize/post-process in Python at eval time; a standalone, from-scratch wrapper for those two steps is a separate, not-yet-built milestone (see main README roadmap).

## Environment notes

Two version-pinning issues surfaced getting here — both are the same underlying category as the numba/CUDA issue from the baseline stage: this ecosystem's GPU-accelerated packages are tightly ABI-pinned to specific CUDA/cuDNN builds, and a newer/default pip resolution silently breaks rather than erroring clearly.

- **`onnxruntime-gpu`** — `CUDAExecutionProvider` wasn't available with the default-resolved version. Fixed by installing **cuDNN v8.9.7 (built for CUDA 11.x)** and downgrading to **`onnxruntime-gpu==1.16.3`**. Confirm with:
  ```python
  import onnxruntime as ort
  print(ort.get_available_providers())  # must include 'CUDAExecutionProvider'
  ```
- **MMDeploy `tools/test.py`** — the `--metrics` CLI flag referenced in MMDeploy's own versioned docs (1.3.1) no longer exists on the current script. Evaluation is now driven entirely by the `test_evaluator` (`KittiMetric`) defined in the model config — same refactor pattern MMDetection3D's own `tools/test.py` went through earlier (which dropped `--eval` for the same reason). Don't trust versioned docs blindly here; check the actual installed script's `--help` output, or the source on GitHub `main`, if a documented flag gets rejected.

## Commands

**1. Export to ONNX** (ONNX Runtime backend — deliberately not TensorRT yet, to isolate export correctness from backend-specific issues):

```bash
python tools/deploy.py \
    configs/mmdet3d/voxel-detection/voxel-detection_onnxruntime_dynamic.py \
    ../mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py \
    ../mmdetection3d/checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth \
    ../mmdetection3d/data/kitti/training/velodyne/000042.bin \
    --work-dir mmdeploy_model/pointpillars \
    --device cuda:0 \
    --show
```

Produces `mmdeploy_model/pointpillars/end2end.onnx`. Uses frame `000042` deliberately — the same frame already hand-verified against ground truth in the baseline stage.

**2. Validate against the full KITTI val split:**

```bash
python tools/test.py \
    configs/mmdet3d/voxel-detection/voxel-detection_onnxruntime_dynamic.py \
    ../mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py \
    --model mmdeploy_model/pointpillars/end2end.onnx \
    --device cuda:0 \
    2>&1 | tee results/onnx_export_eval_log.txt
```

## Result

| | PyTorch (baseline) | ONNX Runtime (exported) | Diff |
|---|---|---|---|
| AP40, Overall, Moderate, 3D | 64.3590 | 64.3536 | 0.0054 |

Effectively a rounding-level difference — an order of magnitude tighter than the baseline-vs-published-checkpoint gap (0.3) confirmed earlier. Confirms the exported ONNX graph (backbone + detection head) preserves the PyTorch model's numerical behavior. Full log: [`results/onnx_export_eval_log.txt`](../results/onnx_export_eval_log.txt).

**What this does *not* validate:** the standalone voxelize/NMS wrapper — this test still relies on MMDeploy's own Python-side pre/post-processing, not a from-scratch implementation. That remains the project's core unsolved milestone.

## Next

TensorRT engine build (FP32 → FP16) via the TensorRT 10.12 Python API, using this validated ONNX graph as input.