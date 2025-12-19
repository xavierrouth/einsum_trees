# Global Data Layout Optimization over Einsum Operators

## Abstract

Einsum (Einstein summation) notation provides a compact and expressive representation for a wide variety of tensor operations, including matrix multiplication, batched operations, contractions, and reductions. However, naive evaluation of multi-operand einsum expressions can lead to suboptimal performance due to poor data layout choices and inefficient contraction orderings. This work implements the Einsum Tree intermediate representation (IR) and associated optimization algorithms to enable global data layout optimization for einsum operators. By representing einsum expressions as binary contraction trees and applying layout reordering transformations, we can generate code that better utilizes hardware primitives such as GEMM operations while minimizing expensive transpose operations.

## 1. Problem Statement and Motivation

Deep neural networks and scientific computing workloads increasingly rely on complex tensor operations that extend beyond simple matrix multiplication. The einsum notation, popularized by NumPy and adopted by frameworks like PyTorch, TensorFlow, and JAX, provides a powerful abstraction for expressing these operations. A single einsum expression like `"pqr,psq,sq,abc,bcp->psr"` can represent a complex multi-tensor contraction that would otherwise require multiple explicit operations.

However, the flexibility of einsum notation creates significant challenges for efficient execution:

1. **Layout Mismatch**: Different tensor operations prefer different data layouts. Matrix multiplication (GEMM) operations expect specific dimension orderings (e.g., row-major vs. column-major), while the input tensors may not conform to these expectations. When dimensions are not properly aligned, compilers must insert expensive transpose operations that can dominate execution time.

2. **Contraction Order Sensitivity**: Multi-operand einsum expressions can be evaluated in many different orders, each with vastly different computational complexity. The order of contractions affects intermediate tensor sizes and the total number of floating-point operations required.

3. **Suboptimal Lowering**: Current compiler implementations often lower einsum operations naively, treating each contraction independently without considering the global layout implications. This local optimization approach misses opportunities for joint optimization across the entire expression.

4. **Hardware Utilization**: Modern accelerators (GPUs, TPUs) achieve peak performance through highly optimized GEMM kernels. However, these kernels have strict requirements on input data layouts. Failure to align tensor layouts with these requirements results in either runtime transposes or suboptimal kernel selection.

The goal of this work is to implement a global layout optimization framework that analyzes entire einsum expressions, determines optimal data layouts for all intermediate tensors, and generates code that minimizes transpose overhead while maximizing utilization of efficient hardware primitives.

## 2. Brief Summary of Existing Work

**Einsum Tree IR [Ding et al., ASPLOS 2024]**: The primary inspiration for this work is the Einsum Tree IR, which provides a systematic approach to einsum optimization. The key insight is representing multi-operand einsum expressions as binary contraction trees, where each internal node represents a binary einsum operation and leaves represent input tensors. The paper introduces a dimension classification scheme (M, N, K, C, R types) that enables systematic layout optimization to minimize transposes. We implement and extend this approach in our framework.

**opt_einsum [Smith & Gray, 2018]**: This library provides algorithms for finding optimal contraction paths for einsum expressions. It uses dynamic programming and greedy heuristics to minimize the total computational cost (FLOPs) of evaluating multi-operand expressions. We use opt_einsum to determine the initial contraction tree structure before applying layout optimizations.

**TASO [Jia et al., SOSP 2019]**: The Tensor Algebra SuperOptimizer uses automated search to discover optimized computation graphs for neural networks. TASO explores equivalent graph transformations and uses cost models to select efficient implementations. While TASO operates at the graph level, our work focuses specifically on intra-operator layout optimization for einsum.

**XLA (Accelerated Linear Algebra)**: Google's XLA compiler performs extensive optimizations on tensor computations, including operation fusion and layout assignment. XLA's layout assignment pass determines memory layouts for tensors based on hardware constraints. Our work can be viewed as a specialized front-end optimization that complements XLA's layout assignment.

**ONNX Runtime Optimizations**: ONNX Runtime includes graph optimizations and operator fusion passes. However, its einsum handling is relatively straightforward, making it an ideal target for demonstrating the benefits of our layout optimization approach.

**cuBLAS and MKL**: These vendor libraries provide highly optimized GEMM implementations that achieve near-peak hardware utilization. Our optimization framework specifically targets generating code that can leverage these libraries by ensuring proper dimension alignment.

## 3. High-Level Overview of Algorithm and Design

### 3.1 Einsum Tree Intermediate Representation

The core of our approach is the **Einsum Tree IR**, a hierarchical representation of einsum expressions. The tree structure is defined as follows:

```
TreeNode:
  - name: identifier for the node's output
  - einsum_string: the einsum equation (e.g., "ij,jk->ik")
  - lhs: left child TreeNode (or None for leaves)
  - rhs: right child TreeNode (or None for leaves)
```

**Leaf nodes** represent input tensors and contain only the tensor's index string (e.g., `"ij"` for a 2D matrix with indices i and j).

**Internal nodes** represent binary contractions, with the einsum_string specifying how the left and right children are combined to produce the output.

For example, the einsum expression `"cigj,iaje,dh,bf,dcba->hgfei"` is decomposed into a tree where:
- Leaves: `cigj`, `iaje`, `dh`, `bf`, `dcba`
- Internal nodes combine pairs of tensors according to an optimal contraction path

### 3.2 Contraction Path Determination

Given a multi-operand einsum expression, we first determine an efficient contraction order using the `opt_einsum` library. The `contraction()` function parses the einsum string and uses `opt_einsum.contract_path()` with the 'eager' optimization strategy to find a good contraction sequence.

```python
def contraction(einsum: str) -> TreeNode:
    path = opt_einsum.contract_path(einsum, *shapes, optimize='eager', shapes=True)
    # Build tree from contraction path
    for (contraction, _, contraction_string, _, _) in path[1].contraction_list:
        # Create internal TreeNode for each binary contraction
        ...
    return root
```

The result is a binary tree where each internal node represents a single binary einsum operation, and the tree structure encodes the contraction order.

### 3.3 Dimension Classification

For each binary einsum node, we classify dimensions into five categories based on their role in the computation:

| Type | Description | Location |
|------|-------------|----------|
| **M** | Row dimension | Left input only, in output |
| **N** | Column dimension | Right input only, in output |
| **K** | Contraction dimension | Both inputs, not in output |
| **C** | Batch/common dimension | Both inputs, in output |
| **R** | Reduction dimension | One input only, not in output |

This classification is implemented in `classify_binary_dimensions()`:

```python
def classify_binary_dimensions(einsum_string: str) -> dict[str, str]:
    lhs, rhs = einsum_string.split("->")
    in1, in2 = lhs.split(",")
    for d in dims:
        count = sum(d in inp for inp in [in1, in2])
        if count == 1 and d not in rhs:
            dim_classification[d] = "R"
        elif count >= 2 and d in rhs:
            dim_classification[d] = "C"
        elif count == 1 and d in rhs:
            dim_classification[d] = "M" if d in in1 else "N"
        elif count >= 2 and d not in rhs:
            dim_classification[d] = "K"
    return dim_classification
```

### 3.4 Layout Optimization Algorithm

The optimization algorithm traverses the tree bottom-up, reordering dimensions at each node to maximize GEMM compatibility. The key insight is that GEMM operations expect inputs in specific layouts:
- Left matrix: dimensions ordered as [..., K, M]
- Right matrix: dimensions ordered as [..., N, K]
- Output: dimensions ordered as [..., M, N]

The `optimize_tree()` function implements this logic:

1. **Parse output dimensions**: Starting from the rightmost dimension of the output, parse and classify dimensions.

2. **Child swapping**: If the rightmost non-batch dimension is of type N instead of M, swap the children of the node. This ensures the "row" dimension comes from the left child.

3. **Dimension reordering**: For each child:
   - Left child: reorder to have rightmost dimensions as K + M (+ C if applicable)
   - Right child: reorder to have rightmost dimensions as N + K (+ C if applicable)

4. **Recursive optimization**: Apply the optimization recursively to child nodes, propagating layout constraints down the tree.

5. **Leaf permutation**: For leaf nodes, insert explicit transpose operations (represented as unary einsum nodes) to achieve the required layout.

### 3.5 Code Generation

The optimized tree is lowered to ONNX format through the `tree_codegen.py` module. The `lower_einsum_node()` function:

1. Parses the original einsum string and builds the tree
2. Optionally applies the `optimize_tree()` transformation
3. Recursively lowers the tree to ONNX nodes:
   - Leaf nodes map to input tensors
   - Unary einsum nodes (transposes) become Einsum or Transpose operations
   - Binary einsum nodes become binary Einsum operations

```python
def lower_einsum_node(graph, model, einsum_node, optimize=False):
    tree_node = contraction(einsum_string)
    if optimize:
        tree_node = optimize_tree(tree_node)
    
    def lower(node, graph):
        # Recursively lower children and create ONNX nodes
        ...
    
    return lower(tree_node, graph)
```

### 3.6 Integration with ONNX Runtime

The main optimization pipeline (`main.py`) provides:

1. Loading of ONNX models containing Einsum operators
2. Application of the layout optimization pass
3. Saving of both non-optimized (tree decomposition only) and fully optimized models
4. Comparison utilities for correctness verification and performance benchmarking

### 3.7 Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Input ONNX Model                      │
│              (with Einsum operators)                     │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Einsum Tree Construction                    │
│   • Parse einsum string                                  │
│   • Compute optimal contraction path (opt_einsum)        │
│   • Build binary tree representation                     │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Layout Optimization                         │
│   • Classify dimensions (M, N, K, C, R)                  │
│   • Determine optimal child orderings                    │
│   • Propagate layout constraints                         │
│   • Insert transpose operations at leaves                │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Code Generation                             │
│   • Lower tree to ONNX binary einsums                    │
│   • Generate transpose operations                        │
│   • Connect inputs/outputs                               │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Output ONNX Model                           │
│        (optimized binary Einsum operators)               │
└─────────────────────────────────────────────────────────┘
```

## 4. Implementation and Evaluation

### 4.1 Implementation

Our implementation consists of the following major components:

**Core Data Structures** ([einsum_tree.py](../src/einsum_tree.py)):
- `TreeNode`: Binary tree representation of einsum expressions
- `classify_binary_dimensions()`: Dimension classification logic
- `optimize_tree()`: Layout optimization algorithm
- `contraction()`: Tree construction from einsum strings

**Code Generation** ([tree_codegen.py](../src/tree_codegen.py)):
- `lower_einsum_node()`: Converts optimized trees to ONNX IR nodes

**Optimization Pipeline** ([main.py](../src/main.py)):
- `lower_einsums()`: Main optimization pass over ONNX models
- Command-line interface for model optimization

**Evaluation Scripts** ([compare_einsum_models.py](../src/compare_einsum_models.py)):
- Correctness verification against NumPy reference
- Performance benchmarking utilities

### 4.2 Experimental Setup

*(Results pending)*

We plan to evaluate on the following benchmarks from the Einsum Tree IR paper:
- Simple matrix multiplication (`ih,bi->bh`)
- Batched matrix multiplication (`pqr,psq->psr`)
- Multi-operand contractions (`pqr,psq,sq,abc,bcp->psr`)

Evaluation will compare:
1. Unoptimized (baseline) vs. optimized einsum via ONNX Runtime (CPU)
2. Unoptimized vs. optimized einsum lowered via JAX + XLA (GPU)
3. Optimized einsum via default XLA vs. custom XLA einsum lowering

### 4.3 Results

*(Experimental results to be added)*

## 5. Conclusions and Future Work

We have implemented the Einsum Tree IR and associated layout optimization algorithms for global optimization of einsum operators. The implementation demonstrates the key concepts of dimension classification, layout propagation, and GEMM-compatible code generation.

Future work includes:
- Integration with XLA's HLO representation for GPU execution
- Exploration of packed GEMM operations for batch dimensions
- Cost model development for layout decision making
- Support for dynamic tensor shapes

## References

1. Ding, Y., et al. "Einsum Tree IR: A Unified Intermediate Representation for Einsum Optimization." ASPLOS 2024.

2. Smith, D. G. A., and Gray, J. "opt_einsum: A Python package for optimizing contraction order for einsum-like expressions." Journal of Open Source Software, 2018.

3. Jia, Z., et al. "TASO: Optimizing Deep Learning Computation with Automatic Generation of Graph Substitutions." SOSP 2019.

4. XLA: Optimizing Compiler for Machine Learning. https://www.tensorflow.org/xla

5. ONNX Runtime. https://onnxruntime.ai/
