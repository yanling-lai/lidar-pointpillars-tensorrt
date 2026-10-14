import argparse
from pathlib import Path
import tensorrt as trt

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "outputs"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def inspect_engine_io(engine_path: str):
    """
    Prints every I/O tensor's name, shape, and dtype.

    Args:
        engine_path (str)
    """
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    print(f"Engine has {engine.num_io_tensors} I/O tensors:")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        mode = engine.get_tensor_mode(name)
        print(f"  [{name}][{mode}]: shape={shape}, dtype={dtype}")

def main():
    parser = argparse.ArgumentParser(description="Inference engine.")
    parser.add_argument("model", help="Model's name")
    parser.add_argument("-f", "--fp16", action="store_true", help="Inference fp16 if set, fp32 otherwise.")
    args = parser.parse_args()

    mode = "fp16" if args.fp16 else "fp32"
    engine_path = OUTPUTS_DIR / "trt_engines" / args.model / f"{args.model}_{mode}.engine"
    inspect_engine_io(engine_path)

if __name__ == "__main__":
    main()