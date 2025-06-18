"""Utilities to build BPQM subcircuits used by the decoders."""

from typing import Dict, List, Tuple, Optional

import networkx as nx
import numpy as np
import qiskit
from qiskit.circuit.library import UCRYGate

# Use the actual Qiskit QuantumCircuit
QuantumCircuit = qiskit.QuantumCircuit


def combine_variable(
    qc: QuantumCircuit,
    idx1: int,
    angles1: List[Tuple[float, Dict[int, int]]],
    idx2: int,
    angles2: List[Tuple[float, Dict[int, int]]]
) -> Tuple[int, List[Tuple[float, Dict[int, int]]]]:
    """Combine two variable nodes of the computation tree."""
    
    # Accumulate output angles for the merged subtree
    angles_out: List[Tuple[float, Dict[int, int]]] = []

    # 1) Gather all original control-qubit indices
    ctrl_orig = []
    if angles1:
        ctrl_orig += list(angles1[0][1].keys())
    if angles2:
        ctrl_orig += list(angles2[0][1].keys())
    # unique preserve order
    ctrl_orig = [bit for bit in dict.fromkeys(ctrl_orig) if bit != idx2]
    # maintain original control ordering
    control_qubits = [bit for bit in ctrl_orig]

    # 2) Prepare lookup arrays
    n_ctrl = len(control_qubits)
    angles_alpha = [None] * (2**n_ctrl)
    angles_beta  = [None] * (2**n_ctrl)

    # 3) Compute α/β for each conditioning
    for t1, c1 in angles1:
        for t2, c2 in angles2:
            # merge control mappings
            orig_controls = {**c1, **c2}
            controls = {bit: val for bit, val in orig_controls.items()}
            angles_out.append((np.arccos(np.cos(t1)*np.cos(t2)), controls))

            # index into multiplex array using original order
            idx_bin = 0
            for bit in ctrl_orig:
                idx_bin = (idx_bin << 1) | orig_controls.get(bit, 0)

            a_min = (
                np.cos(0.5*(t1-t2)) - np.cos(0.5*(t1+t2))
            ) / (np.sqrt(2)*np.sqrt(1 + np.cos(t1)*np.cos(t2)))
            b_min = (
                np.sin(0.5*(t1+t2)) + np.sin(0.5*(t1-t2))
            ) / (np.sqrt(2)*np.sqrt(1 - np.cos(t1)*np.cos(t2)))
            alpha = np.arccos(-a_min) + np.arccos(-b_min)
            beta  = np.arccos(-a_min) - np.arccos(-b_min)

            angles_alpha[idx_bin] = alpha
            angles_beta[idx_bin]  = beta

    # 4) Variable-node gadget with offset indices
    qc.cx(idx2, idx1)
    qc.x(idx1)
    qc.cx(idx1, idx2)
    qc.x(idx1)

    # 5) Reverse controls to match UCRY ordering
    reversed_ctrls = list(reversed(control_qubits))
    # 6) Append uniformly-controlled Ry's
    qc.append(UCRYGate(angles_alpha), [idx2] + reversed_ctrls)
    qc.cx(idx1, idx2)
    qc.append(UCRYGate(angles_beta),  [idx2] + reversed_ctrls)
    qc.cx(idx1, idx2)

    return idx1, angles_out


def combine_check(
    qc: QuantumCircuit,
    idx1: int,
    angles1: List[Tuple[float, Dict[int, int]]],
    idx2: int,
    angles2: List[Tuple[float, Dict[int, int]]],
    check_id: Optional[int] = None,
) -> Tuple[int, List[Tuple[float, Dict[int, int]]]]:
    """Combine two check nodes and optionally apply a syndrome phase."""
    angles_out: List[Tuple[float, Dict[int, int]]] = []
    if check_id is not None:
        qc.cz(check_id, idx1)
    qc.cx(idx1, idx2)

    for t1, c1 in angles1:
        for t2, c2 in angles2:
            orig_controls = {**c1, **c2}
            # branch outputs
            tout_0 = np.arccos((np.cos(t1) + np.cos(t2)) / (1. + np.cos(t1)*np.cos(t2)))
            tout_1 = np.arccos((np.cos(t1) - np.cos(t2)) / (1. - np.cos(t1)*np.cos(t2)))

            # map controls depending on the branch outcome
            ctrl0 = {bit: val for bit, val in orig_controls.items()}
            ctrl0[idx2] = 0
            ctrl1 = {bit: val for bit, val in orig_controls.items()}
            ctrl1[idx2] = 1

            angles_out.append((tout_0, ctrl0))
            angles_out.append((tout_1, ctrl1))

    return idx1, angles_out


def tree_bpqm(
    tree: nx.DiGraph,
    qc: QuantumCircuit,
    root: str,
    offset: int = 0
) -> Tuple[int, List[Tuple[float, Dict[int, int]]]]:
    """Recursively build a BPQM circuit for ``tree`` rooted at ``root``, applying ``offset`` to all indices."""
    succs = list(tree.successors(root))
    # leaf
    if not succs:
        leaf_idx = tree.nodes[root]["qubit_idx"] + offset
        return leaf_idx, tree.nodes[root]["angle"]
    # single child
    if len(succs) == 1:
        idx_child, angles_child = tree_bpqm(tree, qc, succs[0], offset=offset)
        if tree.nodes[root]["type"] == "check":
            check_id = tree.nodes[root].get("check_idx")
            if check_id is not None:
                qc.cz(check_id, idx_child)
        return idx_child, angles_child

    # combine children
    idx, angles = tree_bpqm(tree, qc, succs[0], offset=offset)
    for child in succs[1:]:
        idx2, angles2 = tree_bpqm(tree, qc, child, offset=offset)
        ntype = tree.nodes[root]["type"]
        if ntype == "variable":
            if idx == idx2: # temporary fix
                continue
            idx, angles = combine_variable(qc, idx, angles, idx2, angles2)
        elif ntype == "check":
            check_id = tree.nodes[root].get("check_idx")
            idx, angles = combine_check(qc, idx, angles, idx2, angles2, check_id)
        else:
            raise ValueError(f"Unknown node type '{ntype}'")
    return idx, angles
