# Transform a single einsum expression into an onnx gemm
import onnx
import onnxruntime as ort
from onnxscript import script
from onnxscript import opset15 as op
from onnx import NodeProto
import onnx_ir as ir
import numpy as np
from copy import deepcopy
from onnx import helper, TensorProto

#
# that all tensors are stored contiguously using general row-major
# storage. For example, suppose a three-dimensional tensor
# has dimensions 𝑝, 𝑞, and 𝑟 of sizes 2, 3, and 4, respectively.
# Then the tensor holds a total of 2 · 3 · 4 = 24 scalar values
# contiguously stored in memory. Here, we assume that if the
# tensor is indexed as 𝑝𝑞𝑟 , then the dimension 𝑟 has stride 1,
# 𝑞 has stride 4, and dimension 𝑝 has stride 3 · 4 = 12.
INT_MAX = np.iinfo(np.int32).max
# output an onnx node, that is one of:
# - a single Gemm node
# - a Transpose
# - a packed Gemm
def count_char(s: str, ch: str) -> int:
    return s.count(ch)

def get_dimension_sizes(model, einsum_node: NodeProto) -> dict[str, int]:
    sizes = {}
    # for input_name in einsum_node.input:
    return

# No loops, only gemm / packed gemm
def create_p_unp_gemm(graph: ir.Graph, einsum_node: ir.Node, dim_classification, labeling, dim_sizes: dict[str, int]) -> NodeProto:
    einsum_string = einsum_node.attributes['equation'].value

    lhs, rhs = einsum_string.split("->")
    inputs = lhs.split(",")
    # 𝑝𝑞𝑟, 𝑝𝑠𝑞 → 𝑝𝑠𝑟, need to  swap inputs so  dimensions align for matmul
    # depends on M vs N labeling probably?
    inputs = [list(map(str, inp)) for inp in inputs]
    rhs = list(map(str, rhs))

    # create a new value for the output



    breakpoint()
    node = ir.node(
        graph=graph,
        op_type="Gemm",
        inputs=einsum_node.inputs,
        attributes={
            "transA": 1,
            "transB": 1,
        },
        name="Gemm_from_Einsum: " + einsum_string
    )
    # FIXME: Need to transpose the correct output dimensions
    transpose = ir.node(
        graph=graph,
        op_type="Transpose",
        inputs=[node.outputs[0]],
        name="Transpose_after_Gemm"
    )
    return transpose

def create_gemm_loop_node(graph: ir.Graph, einsum_node: ir.Node, dim_classification, labeling, dim_sizes: dict[str, int]) -> NodeProto:
    breakpoint()
    einsum_string = einsum_node.attributes['equation'].value

    lhs, rhs = einsum_string.split("->")
    inputs = lhs.split(",")

    inputs = [list(map(str, inp)) for inp in inputs]
    rhs = list(map(str, rhs))

    # node = ir.node()
    # assume loops are static for now
    stack = None
    depth = 0
    breakpoint()
    starts_const = ir.from_proto(helper.make_node(
        'Constant',
        inputs=[],
        outputs=['starts'],
        value=helper.make_tensor(
            name='starts_val',
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[0]
        )
    ))

    graph.append(starts_const)
    ends_const = ir.from_proto(helper.make_node(
        'Constant',
        inputs=[],
        outputs=['ends'],
        value=helper.make_tensor(
            name='ends_val',
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[INT_MAX]
        )
    ))
    graph.append(ends_const)
    
    # int_zero = ir.val(graph=graph, name="int_zero", dtype=ir.DataType.INT32, shape=[])
    # int_max = ir.val(graph=graph, name="int_max", dtype=ir.DataType.INT32, shape=[])
    for loop in [d for d, l in labeling.items() if l == "LOOP"]:
        # create loop structure
        if depth == 1:
            # error
            raise NotImplementedError("Nested loops not yet supported")
        if dim_sizes[loop] is None:
            raise ValueError("Dynamic loop sizes not yet supported")
        outputs = []
        # Create an empty sequence
        sequence = ir.node(graph=graph, op_type="SequenceEmpty", inputs=[], name="Empty_sequence_loop_" + str(loop))
        # Split the input tensors along the loop dimension
        split_in0 = ir.node(graph=graph, op_type="SplitToSequence", inputs=[einsum_node.inputs[0]],
        attributes={
            "axis": depth,
            "keepdims": 0,
        }, name="Split_in0_loop_" + str(loop),
        )

        split_in1 = ir.node(graph=graph, op_type="SplitToSequence", inputs=[einsum_node.inputs[1]],
        attributes={
            "axis": depth,
            "keepdims": 0,
        }, name="Split_in1_loop_" + str(loop),
        )
        breakpoint()
        for a in range(dim_sizes[loop]):
            # breakpoint()
            # Slice the input tensors along the loop dimension
            # slice_in0 = ir.node(graph=graph, op_type="Slice", inputs=[
            #     einsum_node.inputs[0],
            #     starts_const.outputs[0],
            #     ends_const.outputs[0],
            #     starts_const.outputs[0] #  fixme this should be depth
            # ], name="Slice_in0" + str(a))
            # slice_in1 = ir.node(graph=graph, op_type="Slice", inputs=[
            #     einsum_node.inputs[1],
            #     starts_const.outputs[0],
            #     ends_const.outputs[0],
            #     starts_const.outputs[0] # fixme this should be depth
            #     ], name="Slice_in1" + str(a))
            # squeeze the sliced inputs
            # squeeze_in0 = ir.node(graph=graph, op_type="Squeeze", inputs=[split_in0.outputs[a], starts_const.outputs[0]], 
            # name="Squeeze_in0" + str(a))
            # squeeze_in1 = ir.node(graph=graph, op_type="Squeeze", inputs=[split_in1.outputs[a], starts_const.outputs[0]], 
            # name="Squeeze_in1" + str(a))
            # Use sequenceAt to get the sliced inputs
            index_val = ir.from_proto(helper.make_node(
                'Constant',
                inputs=[],
                outputs=['index_val_' + str(a)],
                value=helper.make_tensor(
                    name='index_val_ + str(a)',
                    data_type=TensorProto.INT64,
                    dims=[],
                    vals=[a]
                )
            ))
            graph.append(index_val)
            squeeze_in0 = ir.node(graph=graph, op_type="SequenceAt", inputs=[split_in0.outputs[0], index_val.outputs[0]],
            name="Squeeze_in0" + str(a))
            squeeze_in1 = ir.node(graph=graph, op_type="SequenceAt", inputs=[split_in1.outputs[0], index_val.outputs[0]],
            name="Squeeze_in1" + str(a))


            # create gemm for the sliced inputs
            gemm = ir.node(
                graph=graph,
                op_type="Gemm",
                inputs=[squeeze_in0.outputs[0], squeeze_in1.outputs[0]],
                attributes={
                    "transA": 1,
                    "transB": 1,
                },
                name="Gemm_from_Einsum: " + einsum_string + f"_loop_{loop}_{a}"
            )
            # FIXME: Need to transpose the correct output dimensions
            transpose = ir.node(
                graph=graph,
                op_type="Transpose",
                inputs=[gemm.outputs[0]],
                name="Transpose_after_Gemm" + f"_loop_{loop}_{a}"
            )
            sequence = ir.node(
                graph=graph,
                op_type="SequenceInsert",
                inputs=[sequence.outputs[0], transpose.outputs[0]],)
            # outputs.append(transpose.outputs[0])
            pass
        # combine outputs along loop dimension, using stack
        stack = ir.node(
            graph=graph,
            op_type="ConcatFromSequence",
            inputs=[sequence.outputs[0]],
            attributes={
                "axis": depth,
                "new_axis": 1,
            },
            name="Stack_loop_outputs" + str(depth)
        )

        depth += 1
        pass
    return stack

def lower_einsum_node(graph: ir.Graph, model, einsum_node: ir.Node) -> ir.Node:
    # dim_sizes = get_dimension_sizes(model, einsum_node)
    # Get dimension sizes
    # breakpoint()
    # parse the einsum string
    if einsum_node.op_type != "Einsum":
        raise ValueError("Input node is not an Einsum node")
    # breakpoint()
    einsum_string = einsum_node.attributes['equation'].value
    # extract dimension sizes from input tensors
    # for i, input_name in enumerate(einsum_node.input):
    # map einsum to onnx operations
    # einsum_string = "pqr,psq->psr"
    # einsum_string = "pqr,spr->sqr"
    print(einsum_string)

    if einsum_string.count(',') != 1:
        raise NotImplementedError("Only binary einsum expressions are supported")

    dim_classification = classify_binary_dimensions(einsum_string)
    print(dim_classification)

    lhs, rhs = einsum_string.split("->")
    inputs = lhs.split(",")

    inputs = [list(map(str, inp)) for inp in inputs]
    rhs = list(map(str, rhs))

    dim_sizes = {}
    for i, j in zip(einsum_node.inputs[0].shape, inputs[0]):
        dim_sizes[j] = i
    for i, j in zip(einsum_node.inputs[1].shape, inputs[1]):
        dim_sizes[j] = i
    for i, j in zip(einsum_node.outputs[0].shape, rhs):
        dim_sizes[j] = i
    print(dim_sizes)


    labeling = {}
    new_output = rhs
    stage = "check_c"

    for char in reversed(rhs):
        print(char)
        if stage == "check_c" and dim_classification[char] == "C" and all(char == inp[-1] for inp in inputs if char in inp):
            print("trailing contraction dimension, can use gemm")
            # remove it from all strings, 
            for inp in inputs:
                if char in inp:
                    inp.remove(char)
            new_output.remove(char)
            labeling[char] = "GEMM"
        else:
            stage = "check_m"
        if stage == "check_m" and dim_classification[char] == "M" and (char == inputs[0][-1]):
            # remove it from all strings, 
            if char in inputs[0]:
                inputs[0].remove(char)
            new_output.remove(char)
            labeling[char] = "GEMM"
        else:
            stage = "check_n"
        if stage == "check_n" and not dim_classification[char] == "N":
            labeling[char] = "LOOP"
            new_output.remove(char)
        else:
            break
    
    rhs = new_output
    new_inputs_0 = deepcopy(inputs[0])
    for char in reversed(inputs[0]):
        # while the dimension does not coincide with the right-most dimension of I2
        if inputs[1][-1] == char:
            break
        else:
            labeling[char] = "LOOP"
            new_inputs_0.remove(char)
    
    inputs[0] = new_inputs_0
    new_inputs_1 = deepcopy(inputs[1])
    for char in reversed(inputs[1]):
        stage = "check_k"
        if stage == "check_k" and dim_classification[char] == "K" and (char == inputs[1][-1]):
            # remove it from inputs
            inputs[0].remove(char)
            new_inputs_1.remove(char)
            labeling[char] = "GEMM"
        else:
            stage = "check_right"
        
        if stage == "check_right" and not char == rhs[-1]:
            labeling[char] = "LOOP"
            new_inputs_1.remove(char)
        else:
            break
    
    inputs[1] = new_inputs_1

    new_rhs = deepcopy(rhs)
    for char in reversed(rhs):
        if not dim_classification[char] == "N" and inputs[1][-1] == char:
            break
        else:
            labeling[char] = "GEMM"
            inputs[1].remove(char)
            new_rhs.remove(char)
    
    rhs = new_rhs
    # label all remaining dimensions as LOOP
    for char in set(einsum_string) - set(labeling.keys()):
        labeling[char] = "LOOP"
    # remove ',' and '->' from labeling
    if ',' in labeling:
        del labeling[',']
    if '-' in labeling:
        del labeling['-']
    if '>' in labeling:
        del labeling['>']
    print("final labeling:", labeling)
    print("dimension classification:", dim_classification)
    breakpoint()

    # if there are any LOOP dimensions, create loop structure
    if any(l == "LOOP" for l in labeling.values()):
        return create_gemm_loop_node(graph, einsum_node, dim_classification, labeling, dim_sizes)
    else:
        # Create a packed / unpacked gemm
        return create_p_unp_gemm(graph, einsum_node, dim_classification, labeling, dim_sizes)

    # FIXME: insert transpose as needed
    # FIXME: fuse dimensions as needed
    # https://github.com/scalable-analyses/einsum_ir/blob/f40c68666f6eccde7ae5fba66e2413c4f511b677/src/basic/binary/ContractionOptimizer.cpp#L76


    # create onnx nodes for each operation

    # For each loop dimension, create a loop structure
    
    pass

def classify_dimensions(einsum_string: str) -> dict[str, str]:
    lhs, rhs = einsum_string.split("->")
    inputs = lhs.split(",")
    dim_classification = {}
    dims = set([char for char in einsum_string if char.isalpha()])
    # dims = set(inputs) | set(rhs) # union
    print(dims)
    # breakpoint()
    for d in dims:
        count = sum(d in inp for inp in inputs)
        if count == 1 and not d in rhs:
            dim_classification[d] = "R"
        elif count >= 2 and d in rhs:
            dim_classification[d] = "C"
        elif count == 1 and d in rhs:
            dim_classification[d] = "M/N"
        elif count >= 2 and not d in rhs:
            dim_classification[d] = "K"
        else:
            print("what?")
        
    return dim_classification

def classify_binary_dimensions(einsum_string: str) -> dict[str, str]:
    lhs, rhs = einsum_string.split("->")
    in1, in2 = lhs.split(",")
    dim_classification = {}
    dims = set([char for char in einsum_string if char.isalpha()])
    # dims = set(inputs) | set(rhs) # union
    print(dims)
    # breakpoint()
    for d in dims:
        count = sum(d in inp for inp in [in1, in2])
        if count == 1 and not d in rhs:
            dim_classification[d] = "R"
        elif count >= 2 and d in rhs:
            dim_classification[d] = "C"
        elif count == 1 and d in rhs:
            if d in in1:
                dim_classification[d] = "M"
            else:
                dim_classification[d] = "N"
        elif count >= 2 and not d in rhs:
            dim_classification[d] = "K"
        else:
            print("what?")
        
    return dim_classification