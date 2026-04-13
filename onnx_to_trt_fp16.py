import sys

import tensorrt as trt

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
ONNX_PATH = r"C:\Users\admin\Downloads\sahithi_bottling_v1.onnx"
ENGINE_PATH = r"C:\Users\admin\Downloads\sahithi_bottling_v1_fp16.engine"

INPUT_NAME = "images"
BATCH_SIZE = 1
IMG_SIZE = 640
WORKSPACE_GB = 2

# ------------------------------------------------------------------
# LOGGER
# ------------------------------------------------------------------
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def help():
    print("""
========================================================
ONNX → TensorRT FP16 Engine Builder
========================================================

This script converts a trained ONNX deep learning model into a highly
optimized TensorRT engine using FP16 precision for faster inference.

It is typically used for deployment on NVIDIA GPUs to achieve
low-latency and high-throughput inference.

--------------------------------------------------------
WORKING FLOW
--------------------------------------------------------

1. Configuration Setup
   - Defines paths for:
       • ONNX model file
       • Output TensorRT engine file
   - Sets model input name, batch size, image size,
     and GPU workspace memory size.

2. TensorRT Logger Initialization
   - Initializes TensorRT logger to report build status,
     warnings, and errors during engine creation.

3. TensorRT Plugin Loading
   - Loads all required TensorRT plugins.
   - Necessary for layers such as YOLO, custom ops, or
     ONNX graph components.

4. Builder & Network Creation
   - Creates a TensorRT Builder.
   - Creates a network using EXPLICIT_BATCH mode.
   - This allows explicit control over batch size and
     dynamic shapes.

5. ONNX Parsing
   - Reads the ONNX file from disk.
   - Parses the model into a TensorRT network.
   - If parsing fails, detailed parser errors are printed.

6. Builder Configuration
   - Enables FP16 precision for faster inference.
   - Allocates GPU workspace memory (in GB) for engine optimization.
   - More workspace allows better layer fusion and optimization.

7. Optimization Profile Creation
   - Defines input tensor shape constraints:
       • Minimum shape
       • Optimal shape
       • Maximum shape
   - Required for EXPLICIT_BATCH mode.
   - In this script, shapes are fixed for batch size = 1.

8. TensorRT Engine Building
   - TensorRT optimizes the network graph.
   - Performs layer fusion, precision calibration,
     and memory optimization.
   - Produces a highly optimized inference engine.

9. Engine Serialization
   - The built engine is serialized into a `.engine` file.
   - This file can be loaded directly for inference
     without rebuilding.

--------------------------------------------------------
USAGE
--------------------------------------------------------

python build_trt_engine.py

(No command-line arguments required.
Configuration is defined inside the script.)

--------------------------------------------------------
OUTPUT
--------------------------------------------------------

• TensorRT Engine File:
  <ENGINE_PATH>

This engine can be used directly for inference using
TensorRT Runtime APIs (Python or C++).

--------------------------------------------------------
NOTES
--------------------------------------------------------

• Requires NVIDIA GPU with FP16 support.
• ONNX model must be valid and compatible with TensorRT.
• TensorRT, CUDA, and cuDNN must be properly installed.
• Workspace size may need adjustment for large models.

========================================================
""")


sys.exit(0)


def build_engine():
    print("[INFO] Loading TensorRT plugins...")
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_GB << 30)

        print("[INFO] Parsing ONNX...")
        with open(ONNX_PATH, "rb") as f:
            if not parser.parse(f.read()):
                print("[ERROR] Failed to parse ONNX")
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                return

        print("[INFO] Creating optimization profile...")
        profile = builder.create_optimization_profile()
        profile.set_shape(
            INPUT_NAME,
            min=(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE),
            opt=(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE),
            max=(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE),
        )
        config.add_optimization_profile(profile)

        print("[INFO] Building TensorRT engine (FP16)...")
        engine = builder.build_engine(network, config)

        if engine is None:
            raise RuntimeError("Engine build failed")

        print("[INFO] Serializing engine...")
        with open(ENGINE_PATH, "wb") as f:
            f.write(engine.serialize())

        print("✅ TensorRT FP16 engine created:", ENGINE_PATH)


if __name__ == "__main__":
    build_engine()
