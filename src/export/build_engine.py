import argparse
from pathlib import Path

import tensorrt as trt

# src/export/build_engine.py -> repo root is two levels up
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "outputs"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path: str, engine_path: str, fp16: bool = False):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    profile.set_shape("voxels", (1, 32, 4), (16000, 32, 4), (40000, 32, 4))
    profile.set_shape("num_points", (1,), (16000,), (40000,))
    profile.set_shape("coors", (1, 4), (16000, 4), (40000, 4))
    config.add_optimization_profile(profile)

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine build failed - check builder log above for the actual error.")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"Engine saved to {engine_path}")

def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engine from onnx file.")
    parser.add_argument("model", help="Model's name")
    parser.add_argument("-f", "--fp16", action="store_true", help="Compile to fp16 if set, fp32 otherwise.")
    args = parser.parse_args()

    fp16 = args.fp16
    mode = "fp16" if fp16 else "fp32"
    input_path = OUTPUTS_DIR / "onnx" / args.model / "end2end.onnx"
    output_path = OUTPUTS_DIR / "trt_engines" / args.model / f"{args.model}_{mode}.engine"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_engine(input_path, output_path, fp16=fp16)

if __name__ == "__main__":
    main()