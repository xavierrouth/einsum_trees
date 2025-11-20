import onnx
import onnxruntime as ort
from onnxscript import script
from onnxscript import opset18 as op
from onnxscript.onnx_types import FLOAT
from pathlib import Path  # added
import os  # added
import numpy as np

#𝑝𝑞𝑟, 𝑠𝑝𝑟 → 𝑠𝑞𝑟
@script()
def PackedMatMul(
    X: FLOAT["p", "q", "r"],
    W: FLOAT["s", "p", "r"],
) -> FLOAT["s", "q", "r"]:
    return op.Einsum(X, W, equation="pqr,spr->sqr")

# get sizes
def sizes():
    return {
        "p": 10,
        "q": 640,
        "r": 128,
        "s": 96,
    }
# generate test tensors
def generate_tensors():
    import numpy as np
    S = sizes()
    p, q, r, s = S["p"], S["q"], S["r"], S["s"]
    np.random.seed(0)
    X = np.random.randn(p, q, r).astype(np.float32)
    W = np.random.randn(s, p, r).astype(np.float32)
    return X, W
# numpy reference implementation
def loop_batch_matmul_np(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    ref = np.einsum("pqr,spr->sqr", X, W)
    return ref

def _export_model():
    model = PackedMatMul.to_model_proto()  # updated to new function
    out_path = Path(__file__).with_suffix(".model")
    onnx.save(model, out_path.as_posix())

_export_model()

