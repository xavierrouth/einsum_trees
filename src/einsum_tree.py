import opt_einsum
import numpy as np
from copy import copy

class TreeNode:
    def __init__(self, name, einsum_string, lhs=None, rhs=None):
        self.name = name
        self.lhs = lhs
        self.rhs = rhs
        self.einsum_string = einsum_string
        self.output = [] 
    
    def children(self) -> list['TreeNode']:
        return [self.lhs, self.rhs]

    def swap_children(self):
        self.lhs, self.rhs = self.rhs, self.lhs
        string = self.einsum_string.split("->")[0].split(",")
        new_lhs = string[1] + "," + string[0]
        new_eq = new_lhs + "->" + self.einsum_string.split("->")[-1]
        self.einsum_string = new_eq
    
    # To be called on leaf nodes, and insert a transpose, return the transposed node
    def permute_leaf(self, perm: str):
        # Should have no children
        if self.lhs or self.rhs:
            raise ValueError("permute_leaf should only be called on leaf nodes")
        # Create a new Einsum node that permutes the dimensions
        input_str = self.einsum_string
        new_einsum_string = input_str + "->" + perm
        new_node = TreeNode(
            name=perm,
            einsum_string=new_einsum_string,
            lhs=self,
            rhs=None,
        )
        return new_node
    def permute_leaf_2_dim(self):
        # Should only have one child
        if self.lhs or self.rhs:
            raise ValueError("permute_leaf should only be called on leaf nodes")
        # Create a new Einsum node that permutes the dimensions
        input_str = self.einsum_string
        if len(input_str) != 2:
            raise ValueError("Leaf einsum string should have exactly 2 dimensions to permute")
        permuted_str = input_str[1] + input_str[0]
        
        new_einsum_string = input_str + "->" + permuted_str
        new_node = Self(
            name=permuted_str,
            einsum_string=new_einsum_string,
            lhs=self,
            rhs=None,
        )
        return new_node

    def value_name(self) -> str:
        # The rightmost part of the einsum string after '->' is the output indices
        return self.name
        # return self.einsum_string.split("->")[-1]h

    def print_tree(self, level=0):
        indent = "  " * level
        print(f"{indent}- {self.name}: {self.einsum_string}")
        if self.lhs:
            self.lhs.print_tree(level + 1)
        if self.rhs:
            self.rhs.print_tree(level + 1)

def generate_sample_tree() -> TreeNode:
    # cigi, iaje -> acegi
    leaf1 = TreeNode(name="cigj", einsum_string="cigj")
    leaf2 = TreeNode(name="iaje", einsum_string="iaje")
    acegi = TreeNode(name="acegi", einsum_string="cigj,iaje->acegi", lhs=leaf1, rhs=leaf2)
    # bf, dcba -> acdf
    leaf3 = TreeNode(name="bf", einsum_string="bf")
    leaf4 = TreeNode(name="dcba", einsum_string="dcba")
    acdf = TreeNode(name="acdf", einsum_string="bf,dcba->acdf", lhs=leaf3, rhs=leaf4)
    
    # dh, acdf-> acfh
    dh = TreeNode(name="dh", einsum_string="dh")
    acfh = TreeNode(name="acfh", einsum_string="acdf,dh->acfh", lhs=acdf, rhs=dh)
    # acegi, acfh -> hgfei
    hgfei = TreeNode(name="hgfei", einsum_string="acegi,acfh->hgfei", lhs=acegi, rhs=acfh)
    return hgfei

def classify_binary_dimensions(einsum_string: str) -> dict[str, str]:
    lhs, rhs = einsum_string.split("->")
    in1, in2 = lhs.split(",")
    dim_classification = {}
    dims = set([char for char in einsum_string if char.isalpha()])
    # dims = set(inputs) | set(rhs) # union
    print(dims)
    # # breakpoint()
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

def optimize_tree(root: TreeNode) -> TreeNode:
    # Placeholder for tree optimization logic
    # We reorder  the index strings at the child ndoes.

    # Let d be the rightmost dimension in output string
    # check that this is a binary einsum
    if root.lhs is None or root.rhs is None:
        # breakpoint()
        return root  # nothing to optimize
    dim_classification = classify_binary_dimensions(root.einsum_string)
    print(f"optimizing node: {root.einsum_string}")
    print(f"Dimension classification: {dim_classification}")
    # breakpoint()

    lhs, out = root.einsum_string.split("->")
    in1, in2 = lhs.split(",")

    d = out[-1]
    c_substring = ""
    m_substring = ""
    n_substring = ""
    # contiguous substring of all type-K dimensions
    k_substring = "".join([ch for ch in (in1 + in2) if dim_classification.get(ch) == "K"])
    # fix this ^
    # delete repeated characters from k_substring, keeping the same order as k_substring
    seen = set()
    k_substring = "".join([ch for ch in k_substring if not (ch in seen or seen.add(ch))])
    # reverse k_susbtring
    k_substring = k_substring[::-1]
    # k_sustring
    # Unique chars in k_substring, same 
    swapped = False
    print(k_substring)
    if dim_classification[d] == "C":
        # breakpoint()
        # Case 1 
        # 1) Parse out from the right up to the first non-C dimension and collect the dimension sin a contiguous index substring
        stage = 1
        for char in reversed(out):
            if stage == 1 and dim_classification[char] == "C":
                c_substring = char + c_substring 
                continue
            elif stage == 1:
                stage = 2
            # If the first non-C type dimension is of type N, then swap the children of the node.
            if stage == 2 and dim_classification[char] == "N":
                # swap the children of the node
                root.swap_children()
                swapped = True

                # dim_classification[char] = "M"  # change classification since we swapped
                dim_classification = classify_binary_dimensions(root.einsum_string)
                stage = 3
                # continue
            # Continue parsing from the right up to the first non-M type dimension, andd collect the dimensions in a contiguous index substring 
            if stage == 3 and dim_classification[char] == "M":
                m_substring = char + m_substring
                continue
            elif stage == 3:
                stage = 4
            # we are at the first non-M type dimension
            # if the first non-M type dimension is not of type N, then continue parsing up the first type N dimension
            if stage == 4 and dim_classification[char] != "N":
                stage = 5
            elif stage == 4:
                stage = 6
            # Continue parsing up to the first N-type dimension
            if stage == 5 and dim_classification[char] != "N":
                continue
            elif stage == 5:
                stage = 6
            # continue parsing from the right up to the first non-type dimension or the end of the string, and collect all dimension in a contiguous substring gN
            if stage == 6 and dim_classification[char] == "N":
                n_substring = char + n_substring
                continue
            else:
                break
        print(f"c_substring: {c_substring}, m_substring: {m_substring}, n_substring: {n_substring}")
        # breakpoint()
        # Reorder in1 such that its rightmost substring is k_substring + m_substring + n_substring
        # Reorder in2 such that its rightmost substring is n_substring + k_substring + c_substring
        # rearrange in1 to have the same order as lhs_permutation, don't add or remove any characters

        lhs_permutation = k_substring + m_substring + c_substring
        # sort it according to the order in lhs_permutation
        rhs_permutation = n_substring + k_substring + c_substring

        if swapped:
            new_lhs = ''.join(sorted(in2, key=lambda x: lhs_permutation.index(x) if x in lhs_permutation else -1))
            new_rhs = ''.join(sorted(in1, key=lambda x: rhs_permutation.index(x) if x in rhs_permutation else -1))
        else:
            new_lhs = ''.join(sorted(in1, key=lambda x: lhs_permutation.index(x) if x in lhs_permutation else -1))
            new_rhs = ''.join(sorted(in2, key=lambda x: rhs_permutation.index(x) if x in rhs_permutation else -1))

    elif dim_classification[d] == "N" or dim_classification[d] == "M":
        # breakpoint()
        if dim_classification[d] == "N":
            # swap children
            root.swap_children()
            # dim_classification[d] = "M"
            swapped = True
            dim_classification = classify_binary_dimensions(root.einsum_string)
        # 
        # Parse out from the right up to the first non-M dimension and collect the dimension sin a contiguous index substring
        stage = 1
        for char in reversed(out):
            if stage == 1 and dim_classification[char] == "M":
                m_substring = char + m_substring
                continue
            elif stage == 1:
                stage = 2
            # we are at the first non-M type dimension
            # if the first non-M type dimension is not of type N, then continue parsing up the first type N dimension
            if stage == 2 and dim_classification[char] != "N":
                stage = 3
            elif stage == 2:
                stage = 4
            # Continue parsing up to the first N-type dimension
            if stage == 3 and dim_classification[char] != "N":
                continue
            elif stage == 3:
                stage = 4
            # continue parsing from the right up to the first non-type dimension or the end of the string, and collect all dimension in a contiguous substring gN
            if stage == 4 and dim_classification[char] == "N":
                n_substring = char + n_substring
                continue
            elif stage == 4:
                # non-N, don't keep going``
                break
        print(f"c_substring: {c_substring}, m_substring: {m_substring}, n_substring: {n_substring}")
        # breakpoint()

        # Reorder in1 such that its rightmost substring is k_substring + m_substring
        # Reorder in2 such that its rightmost substring is n_substring + k_substring
        lhs_permutation = k_substring + m_substring
        rhs_permutation = n_substring + k_substring

        if swapped:
            new_lhs = ''.join(sorted(in2, key=lambda x: lhs_permutation.index(x) if x in lhs_permutation else -1))
            new_rhs = ''.join(sorted(in1, key=lambda x: rhs_permutation.index(x) if x in rhs_permutation else -1))
        else:
            new_lhs = ''.join(sorted(in1, key=lambda x: lhs_permutation.index(x) if x in lhs_permutation else -1))
            new_rhs = ''.join(sorted(in2, key=lambda x: rhs_permutation.index(x) if x in rhs_permutation else -1))
    
    # breakpoint()
    # if swapped:
    #     rhs_permutation, lhs_permutation = new_lhs, new_rhs
    # else:
    lhs_permutation, rhs_permutation = new_lhs, new_rhs
    print(f"lhs_permutation: {lhs_permutation}, rhs_permutation: {rhs_permutation}")
    
    # if the child nodes are leaves, permute them accordingly
    if root.lhs and not root.lhs.lhs and not root.lhs.rhs:
        root.lhs = root.lhs.permute_leaf(lhs_permutation)
    else:
        #Lower the lhs child
        # Set the output of the einsum string of the lhs's child to the new lhs_permutation
        root.lhs.einsum_string = root.lhs.einsum_string.split("->")[0] + "->" + lhs_permutation
        root.lhs = optimize_tree(root.lhs)
    if root.rhs and not root.rhs.lhs and not root.rhs.rhs:
        root.rhs = root.rhs.permute_leaf(rhs_permutation)
    else:
        root.rhs.einsum_string = root.rhs.einsum_string.split("->")[0] + "->" + rhs_permutation
        root.rhs = optimize_tree(root.rhs)
    # Repair einsum string at this node
    new_lhs = root.lhs.einsum_string.split("->")[-1]
    new_rhs = root.rhs.einsum_string.split("->")[-1]
    new_einsum_string = new_lhs + "," + new_rhs + "->" + out
    root.einsum_string = new_einsum_string
    return root

def contraction(einsum: str) -> TreeNode:
    p = 10
    q = 64
    r = 128
    s = 32
    a = 10
    b = 20
    c = 30
    operands = einsum.split("->")[0].split(",")

    shapes = []

    for op_str in operands:
        shape = []
        for char in op_str:
            shape.append(10) # arbitrary size
        shapes.append(shape)
    path = opt_einsum.contract_path(einsum, *shapes, optimize='eager', shapes=True)
    # breakpoint()
    print(path)

    # Construct a tree from the contraction path

    # Construct a Node for each base tensor

    nodes_map = {}
    # Base Nodes
    for node in einsum.split("->")[0].split(","):
        nodes_map[node] = TreeNode(name=node, einsum_string=node)

    for i, (contraction, _, contraction_string, _, _) in enumerate(path[1].contraction_list):
        lhs, out = contraction_string.split("->")
        first, second = lhs.split(",")
        node = TreeNode(name=f"node_{i}", einsum_string=contraction_string, lhs=nodes_map[first], rhs=nodes_map[second])
        nodes_map[out] = copy(node)
        
    print("Constructed Tree:")
    root = nodes_map[path[1].contraction_list[-1][2].split("->")[1]]
    # root.print_tree()
    return root

if __name__ == "__main__":
    # contraction("pqr,spr->sqr")
    tree = generate_sample_tree()
    tree.print_tree()
    optimized_tree = optimize_tree(tree)
    print("Optimized Tree:")
    optimized_tree.print_tree()
    # Other tree

    other_tree = contraction("cigj,iaje,dh,bf,abccba->hgfei")
    print("Other Tree:")
    other_tree.print_tree()
    # contraction("pqr,psq,sq,abc,bcp,abcpqr->psr")