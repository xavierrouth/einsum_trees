For benchmarks, we plan to use at least the same set of Einsum expressions as in the Einsum Tree IR paper (table 5), and evaluate them on GPU and CPU.

We will measure runtime of compiled einsum expression. We will not measure compile time. Our evaluation / experiments will look like:

    unoptimized (baseline) vs optimized einsum via ONNX RT (cpu)
    unoptimized (baseline) vs optimized einsum lowered via jax + default XLA (gpu)
    optimized einsum lowered via jax + default XLA vs optimized einsum lowered via jax + custom XLA einsum lowering (gpu)
