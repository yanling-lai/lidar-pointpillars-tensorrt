#!/usr/bin/env bash
set -euo pipefail
cd ./mmdeploy
python3 -W ignore ./tools/deploy.py \
    ./configs/mmdet3d/voxel-detection/voxel-detection_onnxruntime_dynamic.py \
    ../mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py \
    ../mmdetection3d/checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth \
    ../mmdetection3d/data/kitti/training/velodyne/000042.bin \
    --work-dir ../outputs/onnx/pointpillars \
    --device cuda:0 \
    --show
