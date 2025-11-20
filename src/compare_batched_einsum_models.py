import argparse
from pathlib import Path
import numpy as np
import time

try:
    import onnxruntime as ort
    from inputs import batch_matmul as einsum_mod  # triggers export of LoopBatchmatMul model
except Exception as e:
    raise SystemExit(f"Required packages missing: {e}\nInstall: pip install onnxruntime onnx onnxscript")

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
    p = argparse.ArgumentParser(description="Batched matmul comparison for LoopBatchmatMul (pqr,psq->psr).")
    p.add_argument("--batch", type=int, default=10, help="Fixed to 10 (static in exported model).")
    p.add_argument("--q", type=int, default=640)
    p.add_argument("--r", type=int, default=1280)
    p.add_argument("--s", type=int, default=960)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nonopt-path", type=str, default=None)
    p.add_argument("--opt-path", type=str, default=None)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--provider", type=str, default="CPUExecutionProvider")
    args = p.parse_args()

    B, q, r, s = args.batch, args.q, args.r, args.s
    if B != 10:
        raise SystemExit("Error: --batch must be 10 (model exported with static first dim=10).")

    np.random.seed(args.seed)
    X = np.random.randn(B, q, r).astype(np.float32)
    W = np.random.randn(B, s, q).astype(np.float32)

    ref = np.einsum("pqr,psq->psr", X, W)

    default_model = Path(einsum_mod.__file__).with_suffix(".model")
    nonopt_path = Path(args.nonopt_path) if args.nonopt_path else default_model
    opt_path = Path(args.opt_path) if args.opt_path else nonopt_path.parent / "optimized_einsum.model"

    if not nonopt_path.exists():
        raise SystemExit(f"Model not found: {nonopt_path}")

    sess_nonopt = _create_session(nonopt_path, args.provider)
    (y_nonopt,) = sess_nonopt.run(None, {"X": X, "W": W})
    diff_nonopt = float(np.max(np.abs(y_nonopt - ref)))

    if opt_path.exists():
        sess_opt = _create_session(opt_path, args.provider)
        (y_opt,) = sess_opt.run(None, { "X": X, "W": W})
        diff_opt = float(np.max(np.abs(y_opt - ref)))
        diff_between = float(np.max(np.abs(y_nonopt - y_opt)))
    else:
        y_opt = None

    print(f"Shapes: X={X.shape}, W={W.shape}, Ref={ref.shape}")
    print(f"Non-optimized model: {nonopt_path}")
    if y_opt is not None:
        print(f"Optimized model: {opt_path}")
    print(f"Ref checksum: {float(ref.sum()):.6g}")
    print(f"Non-optimized checksum: {float(y_nonopt.sum()):.6g}")
    print(f"Max abs diff (NumPy vs non-optimized): {diff_nonopt:.6g}")
    if y_opt is not None:
        print(f"Optimized checksum: {float(y_opt.sum()):.6g}")
        print(f"Max abs diff (NumPy vs optimized): {diff_opt:.6g}")
        print(f"Max abs diff (non-optimized vs optimized): {diff_between:.6g}")
    else:
        print(f"Optimized model missing; expected at: {opt_path}")

    print(f"Benchmarking provider={args.provider}, warmup={args.warmup}, runs={args.runs}")
    _bench("NumPy batched matmul", lambda: np.einsum("pqr,psq->psr", X, W), args.warmup, args.runs)
    _bench("ORT non-optimized", lambda: sess_nonopt.run(None, {"X": X, "W": W}), args.warmup, args.runs)
    if y_opt is not None:
        sess_opt = _create_session(opt_path, args.provider)
        _bench("ORT optimized", lambda: sess_opt.run(None, {"X": X, "W": W}), args.warmup, args.runs)

if __name__ == "__main__":
    main()
