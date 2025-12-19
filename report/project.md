Global Data Layout Optimization over Einsum Operators:
Tensor operators may prefer certain data layouts, such as dimension order, that maximize performance
compared to other layouts. Multiple operators in the same DNN can have conflicts in layout preference, calling
for a global, joint optimization method. Einsum Tree [2] provides an example of such layout optimization
over einsum operators. (You can opt to bring the Einsum Tree IR to your select tensor compiler if that helps
you implement their optimizations.)