import onnx
import onnxruntime as ort
from onnxscript import script
from onnxscript import opset18 as op
from onnxscript.onnx_types import FLOAT
from pathlib import Path  # added
import os  # added
import numpy as np

@script()
def TTGTMatMul( 
    X: FLOAT["i", "h"],
    W: FLOAT["b", "i"],
) -> FLOAT["b", "h"]:
    return op.Einsum(X, W, equation="ih,bi->bh")
# get sizes
def sizes():
    return {
        "b": 640,
        "i": 128,
        "h": 96,
    }

# numpy test tensor generation
def generate_tensors():
    S = sizes()
    b, i, h = S["b"], S["i"], S["h"]
    np.random.seed(0)
    X = np.random.randn(i, h).astype(np.float32)
    W = np.random.randn(b, i).astype(np.float32)
    return X, W
    
def _export_model():
    breakpoint()
    model = TTGTMatMul.to_model_proto()  # updated to new function
    out_path = Path(__file__).with_suffix(".model")
    onnx.save(model, out_path.as_posix())

# numpy reference implementation
def ttgt_matmul_np(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    ref = np.einsum("ih,bi->bh", X, W)
    return ref
_export_model()

