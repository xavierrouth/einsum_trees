import argparse
from pathlib import Path
import numpy as np

try:
    import onnx
    import onnxruntime as ort
    from inputs.einsum import TTGTMatMul as EinsumMatMul # imported external model
except Exception as e:
    raise SystemExit(
        "Requires: pip install onnx onnxruntime onnxscript\n"
        f"Import error: {e}"
    )

def build_model():
    # model now built from imported TwoMatMuls
    model = EinsumMatMul.to_model_proto()
    return model

def save_model(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path.as_posix())

def run_cpu(model_path: Path, X: np.ndarray, W: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    (out,) = sess.run(None, {"X": X, "W": W})
    return out

def main():
    p = argparse.ArgumentParser(description="ONNXScript two chained MatMul demo")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--in-features", type=int, default=8)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--out-features", type=int, default=5)
    p.add_argument("--model-path", type=str, default=None)
    args = p.parse_args()

    b = args.batch
    i = args.in_features
    h = args.hidden
    o = args.out_features

    np.random.seed(0)
    X = np.random.randn(b, i).astype(np.float32)
    W = np.random.randn(i, h).astype(np.float32)

    model = build_model()
    model_path = Path(args.model_path) if args.model_path else Path(__file__).resolve().parent.parent / "outputs" / "two_matmuls.onnx"
    save_model(model, model_path)

    out = run_cpu(model_path, X.T, W.T).T
    # out = out.T
    # Reference with NumPy
    ref = X @ W
    max_abs_diff = float(np.max(np.abs(out - ref)))
    print(f"Saved model: {model_path}")
    print(f"Shapes: X={X.shape}, W={W.shape}, Y={out.shape}")
    print(f"Max abs diff vs NumPy: {max_abs_diff:.6g}")
    print(f"Output checksum: {float(out.sum()):.6g}")
    print(f"Ref Output checksum: {float(ref.sum()):.6g}")


if __name__ == "__main__":
    main()
