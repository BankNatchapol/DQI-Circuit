"""
Cirq version of BPQM main logic.
"""

# TODO: Implement Cirq-based BPQM logic here

import cirq

class CirqBPQM:
    def __init__(self):
        pass
    # Add Cirq BPQM methods here 

"""Utilities to build BPQM subcircuits - Cirq version."""

from typing import Dict, List, Tuple, Optional

import networkx as nx
import numpy as np
import cirq


def combine_variable_cirq(
    circuit: cirq.Circuit,
    qubits: List[cirq.Qid],
    idx1: int,
    angles1: List[Tuple[float, Dict[int, int]]],
    idx2: int,
    angles2: List[Tuple[float, Dict[int, int]]]
) -> Tuple[int, List[Tuple[float, Dict[int, int]]]]:
    """Combine two variable nodes of the computation tree - Cirq version."""
    
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
    circuit.append(cirq.CNOT(qubits[idx2], qubits[idx1]))
    circuit.append(cirq.X(qubits[idx1]))
    circuit.append(cirq.CNOT(qubits[idx1], qubits[idx2]))
    circuit.append(cirq.X(qubits[idx1]))

    # 5) Reverse controls to match UCRY ordering
    reversed_ctrls = list(reversed(control_qubits))
    
    # 6) Append uniformly-controlled Ry's using Cirq
    # Create controlled Ry gates for each control configuration
    if control_qubits:
        # Create a controlled rotation gate
        for i in range(2**n_ctrl):
            if angles_alpha[i] is not None:
                # Determine control values from binary representation
                control_vals = []
                temp_i = i
                for _ in range(n_ctrl):
                    control_vals.append(temp_i & 1)
                    temp_i >>= 1
                control_vals.reverse()
                
                # Create controlled Ry gate
                ops = []
                # Add control conditions
                for j, ctrl_idx in enumerate(reversed_ctrls):
                    if control_vals[j] == 0:
                        ops.append(cirq.X(qubits[ctrl_idx]))
                
                # Add controlled rotation
                if n_ctrl == 0:
                    ops.append(cirq.ry(angles_alpha[i])(qubits[idx2]))
                else:
                    ctrl_qubits = [qubits[idx] for idx in reversed_ctrls]
                    ops.append(cirq.ControlledGate(
                        cirq.ry(angles_alpha[i]), 
                        num_controls=n_ctrl
                    )(*ctrl_qubits, qubits[idx2]))
                
                # Remove control conditions
                for j, ctrl_idx in enumerate(reversed_ctrls):
                    if control_vals[j] == 0:
                        ops.append(cirq.X(qubits[ctrl_idx]))
                
                circuit.append(ops)
    else:
        # No controls, just apply the rotation
        if angles_alpha[0] is not None:
            circuit.append(cirq.ry(angles_alpha[0])(qubits[idx2]))
    
    circuit.append(cirq.CNOT(qubits[idx1], qubits[idx2]))
    
    # Similar for beta angles
    if control_qubits:
        for i in range(2**n_ctrl):
            if angles_beta[i] is not None:
                control_vals = []
                temp_i = i
                for _ in range(n_ctrl):
                    control_vals.append(temp_i & 1)
                    temp_i >>= 1
                control_vals.reverse()
                
                ops = []
                for j, ctrl_idx in enumerate(reversed_ctrls):
                    if control_vals[j] == 0:
                        ops.append(cirq.X(qubits[ctrl_idx]))
                
                if n_ctrl == 0:
                    ops.append(cirq.ry(angles_beta[i])(qubits[idx2]))
                else:
                    ctrl_qubits = [qubits[idx] for idx in reversed_ctrls]
                    ops.append(cirq.ControlledGate(
                        cirq.ry(angles_beta[i]), 
                        num_controls=n_ctrl
                    )(*ctrl_qubits, qubits[idx2]))
                
                for j, ctrl_idx in enumerate(reversed_ctrls):
                    if control_vals[j] == 0:
                        ops.append(cirq.X(qubits[ctrl_idx]))
                
                circuit.append(ops)
    else:
        if angles_beta[0] is not None:
            circuit.append(cirq.ry(angles_beta[0])(qubits[idx2]))
    
    circuit.append(cirq.CNOT(qubits[idx1], qubits[idx2]))

    return idx1, angles_out


def combine_check_cirq(
    circuit: cirq.Circuit,
    qubits: List[cirq.Qid],
    idx1: int,
    angles1: List[Tuple[float, Dict[int, int]]],
    idx2: int,
    angles2: List[Tuple[float, Dict[int, int]]],
    check_id: Optional[int] = None,
) -> Tuple[int, List[Tuple[float, Dict[int, int]]]]:
    """Combine two check nodes and optionally apply a syndrome phase - Cirq version."""
    angles_out: List[Tuple[float, Dict[int, int]]] = []
    if check_id is not None:
        circuit.append(cirq.CZ(qubits[check_id], qubits[idx1]))
    circuit.append(cirq.CNOT(qubits[idx1], qubits[idx2]))

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


def tree_bpqm_cirq(
    tree: nx.DiGraph,
    qubits: List[cirq.Qid],
    root: str,
    offset: int = 0
) -> Tuple[cirq.Circuit, int]:
    """Recursively build a BPQM circuit for tree rooted at root - Cirq version.
    
    Returns:
        circuit: The Cirq circuit
        meas_idx: The index of the measurement qubit
    """
    circuit = cirq.Circuit()
    
    def _recurse(node: str) -> Tuple[int, List[Tuple[float, Dict[int, int]]]]:
        succs = list(tree.successors(node))
        
        # leaf
        if not succs:
            leaf_idx = tree.nodes[node]["qubit_idx"]
            if hasattr(leaf_idx, 'x'):  # If it's a Qid object
                leaf_idx = leaf_idx.x
            leaf_idx += offset
            return leaf_idx, tree.nodes[node]["angle"]
        
        # single child
        if len(succs) == 1:
            idx_child, angles_child = _recurse(succs[0])
            if tree.nodes[node]["type"] == "check":
                check_id = tree.nodes[node].get("check_idx")
                if check_id is not None:
                    circuit.append(cirq.CZ(qubits[check_id], qubits[idx_child]))
            return idx_child, angles_child

        # combine children
        idx, angles = _recurse(succs[0])
        for child in succs[1:]:
            idx2, angles2 = _recurse(child)
            ntype = tree.nodes[node]["type"]
            if ntype == "variable":
                if idx == idx2:  # temporary fix
                    continue
                idx, angles = combine_variable_cirq(circuit, qubits, idx, angles, idx2, angles2)
            elif ntype == "check":
                check_id = tree.nodes[node].get("check_idx")
                idx, angles = combine_check_cirq(circuit, qubits, idx, angles, idx2, angles2, check_id)
            else:
                raise ValueError(f"Unknown node type '{ntype}'")
        return idx, angles
    
    meas_idx, _ = _recurse(root)
    return circuit, meas_idx 