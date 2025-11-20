import argparse
from pathlib import Path
import numpy as np
import time  # added

try:
    import onnxruntime as ort
    from inputs import einsum as einsum_mod  # ensures export of einsum.model on import
except Exception as e:
    raise SystemExit(f"Required packages missing: {e}\nInstall: pip install onnxruntime onnx onnxscript")

def run_ttgt(model_path: Path, X: np.ndarray, W: np.ndarray) -> np.ndarray:
    # TTGTMatMul signature: (i,h),(b,i)->(b,h); our X:(b,i), W:(i,h)
    sess = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    (out,) = sess.run(None, {"X": X, "W": W})
    return out

def _create_session(path: Path, provider: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    return ort.InferenceSession(path.as_posix(), sess_options=so, providers=[provider])

def _bench(label: str, fn, warmup: int, runs: int):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    dt = time.perf_counter() - t0
    avg_ms = (dt / max(runs, 1)) * 1000.0
    it_s = runs / dt if dt > 0 else float("inf")
    print(f"{label}: avg {avg_ms:.3f} ms, {it_s:.1f} it/s")

def main():
    p = argparse.ArgumentParser(description="Compare NumPy, non-optimized and optimized Einsum MatMul outputs.")
    p.add_argument("--batch", type=int, default=4000)
    p.add_argument("--in-features", type=int, default=800)
    p.add_argument("--hidden", type=int, default=1600)
    p.add_argument("--nonopt-path", type=str, default=None)
    p.add_argument("--opt-path", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    # --- benchmarking args ---
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--provider", type=str, default="CPUExecutionProvider")
    args = p.parse_args()

    b, i, h = args.batch, args.in_features, args.hidden
    np.random.seed(args.seed)
    X = np.random.randn(i, h).astype(np.float32)
    W = np.random.randn(b, i).astype(np.float32)
    breakpoint()
    ref = (X.T @ W.T).T  # NumPy reference

    # Resolve model paths
    default_nonopt = Path(einsum_mod.__file__).with_suffix(".model")
    print(f"Default non-optimized model path: {default_nonopt}")
    nonopt_path = Path(args.nonopt_path) if args.nonopt_path else default_nonopt
    opt_path = Path(args.opt_path) if args.opt_path else nonopt_path.parent / "optimized_einsum.model"

    if not nonopt_path.exists():
        raise SystemExit(f"Non-optimized model not found: {nonopt_path}")

    y_nonopt = run_ttgt(nonopt_path, X, W)
    max_diff_np_nonopt = float(np.max(np.abs(y_nonopt - ref)))

    if opt_path.exists():
        breakpoint()
        y_opt = run_ttgt(opt_path, X, W)  # removed stray transpose
        max_diff_np_opt = float(np.max(np.abs(y_opt - ref)))
        max_diff_nonopt_opt = float(np.max(np.abs(y_nonopt - y_opt)))
    else:
        y_opt = None

    print(f"Shapes: X={X.shape}, W={W.shape}, Ref={ref.shape}")
    print(f"Non-optimized model: {nonopt_path}")
    if y_opt is not None:
        print(f"Optimized model: {opt_path}")
    print(f"Ref checksum: {float(ref.sum()):.6g}")
    print(f"Non-optimized checksum: {float(y_nonopt.sum()):.6g}")
    print(f"Max abs diff (NumPy vs non-optimized): {max_diff_np_nonopt:.6g}")
    if y_opt is not None:
        print(f"Optimized checksum: {float(y_opt.sum()):.6g}")
        print(f"Max abs diff (NumPy vs optimized): {max_diff_np_opt:.6g}")
        print(f"Max abs diff (non-optimized vs optimized): {max_diff_nonopt_opt:.6g}")
    else:
        print(f"Optimized model missing; skipped comparison. Expected at: {opt_path}")

    # --- benchmarking ---
    print(f"Benchmarking with provider={args.provider}, warmup={args.warmup}, runs={args.runs}")
    feeds = {"X": X, "W": W}
    sess_nonopt = _create_session(nonopt_path, args.provider)
    _bench("NumPy matmul", lambda: (X.T @ W.T).T, args.warmup, args.runs)
    _bench("ORT non-optimized", lambda: sess_nonopt.run(None, feeds), args.warmup, args.runs)
    if y_opt is not None:
        sess_opt = _create_session(opt_path, args.provider)
        _bench("ORT optimized", lambda: sess_opt.run(None, feeds), args.warmup, args.runs)

if __name__ == "__main__":
    main()
