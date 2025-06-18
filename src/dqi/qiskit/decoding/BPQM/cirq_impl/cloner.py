"""Cloning utilities used by the BPQM decoders - Cirq version."""

from typing import Dict, List, Tuple

import numpy as np
import networkx as nx
import cirq


class CirqCloner:
    """Base class for all Cirq cloner implementations."""

    def mark_angles(self, graph: nx.Graph, occurances: Dict[str, int]) -> None:
        """Annotate ``graph`` with angles for cloning."""
        raise NotImplementedError

    def generate_cloner_circuit(
        self, graph: nx.Graph, occurances: Dict[str, int], qubit_mapping: Dict[str, cirq.Qid], n_qubits: int,
    ) -> cirq.Circuit:
        """Return the circuit implementing the required cloning operations."""
        raise NotImplementedError


class CirqVarNodeCloner(CirqCloner):
    """Simple cloner treating all outputs symmetrically - Cirq version."""

    def __init__(self, theta: float) -> None:
        self.theta = theta

    def mark_angles(self, graph: nx.Graph, occurances: Dict[str, int]) -> None:
        leaves = [n for n in graph.nodes() if graph.nodes[n]["type"] == "output"]
        for l in leaves:
            n = occurances[l.split("_")[0].replace("y", "x")]
            t = np.arccos(np.cos(self.theta)**(1./n))
            graph.nodes[l]["angle"] = [(t, {})]

    def cloner_unitary(self, t1: float, t2: float) -> np.ndarray:
        aplus = (1./np.sqrt(2.)) * (np.cos(0.5*(t1-t2)) + np.cos(0.5*(t1+t2))) \
                     / np.sqrt(1. + np.cos(t1)*np.cos(t2))
        amin  = (1./np.sqrt(2.)) * (np.cos(0.5*(t1-t2)) - np.cos(0.5*(t1+t2))) \
                     / np.sqrt(1. + np.cos(t1)*np.cos(t2))
        bplus = (1./np.sqrt(2.)) * (np.sin(0.5*(t1+t2)) - np.sin(0.5*(t1-t2))) \
                     / np.sqrt(1. - np.cos(t1)*np.cos(t2))
        bmin  = (1./np.sqrt(2.)) * (np.sin(0.5*(t1+t2)) + np.sin(0.5*(t1-t2))) \
                     / np.sqrt(1. - np.cos(t1)*np.cos(t2))
        return np.array([
            [aplus, 0.,    0.,     amin   ],
            [amin,  0.,    0.,     -aplus ],
            [0.,    bplus, bmin,   0.     ],
            [0.,    bmin,  -bplus, 0.     ]
        ]).T

    def generate_cloner_circuit(
        self,
        graph: nx.Graph,
        occurances: Dict[str, int],
        qubit_mapping: Dict[str, cirq.Qid],
        n_qubits: int,
    ) -> cirq.Circuit:
        n = len(occurances)
        circuit = cirq.Circuit()
        
        for i in range(n):
            if occurances[f"x{i}"] <= 1:
                continue
            elif occurances[f"x{i}"] == 2:
                theta_out = np.arccos(np.cos(self.theta)**(1./2.))
                matrix = self.cloner_unitary(theta_out, theta_out)
                gate = cirq.MatrixGate(matrix)
                circuit.append(gate(qubit_mapping[f"y{i}_1"], qubit_mapping[f"x{i}_0"]))
            elif occurances[f"x{i}"] == 3:
                theta_mid = np.arccos(np.cos(self.theta)**(2./3.))
                theta_out = np.arccos(np.cos(self.theta)**(1./3.))
                matrix1 = self.cloner_unitary(theta_out, theta_mid)
                matrix2 = self.cloner_unitary(theta_out, theta_out)
                gate1 = cirq.MatrixGate(matrix1)
                gate2 = cirq.MatrixGate(matrix2)
                circuit.append(gate1(qubit_mapping[f"y{i}_1"], qubit_mapping[f"x{i}_0"]))
                circuit.append(gate2(qubit_mapping[f"y{i}_2"], qubit_mapping[f"y{i}_1"]))
            else:
                raise Exception("cloning a qubit to >3 output qubits not yet implemented")
        return circuit


class CirqExtendedVarNodeCloner(CirqCloner):
    """Equatorial (phase-covariant) cloner for variable nodes - Cirq version."""

    def __init__(self, theta: float) -> None:
        """
        Args:
            theta: Bloch-sphere polar angle of the input state.
        """
        self.theta = theta

    def mark_angles(self, graph: nx.Graph, occurances: Dict[str, int]) -> None:
        """Mark each output node with its required rotation angle."""
        for node, data in graph.nodes(data=True):
            if data.get("type") == "output":
                var = node.split("_")[0].replace("y", "x")
                M = occurances[var]
                theta_out = np.arccos(np.cos(self.theta) ** (1.0 / M))
                data["angle"] = [(theta_out, {})]

    def cloner_unitary(self, t_out: float, t_prev: float) -> np.ndarray:
        """Construct the 4×4 two-qubit cloning unitary."""
        aplus = (1/np.sqrt(2)) * (np.cos(0.5*(t_out - t_prev)) + np.cos(0.5*(t_out + t_prev))) \
                / np.sqrt(1 + np.cos(t_out)*np.cos(t_prev))
        amin  = (1/np.sqrt(2)) * (np.cos(0.5*(t_out - t_prev)) - np.cos(0.5*(t_out + t_prev))) \
                / np.sqrt(1 + np.cos(t_out)*np.cos(t_prev))
        bplus = (1/np.sqrt(2)) * (np.sin(0.5*(t_out + t_prev)) - np.sin(0.5*(t_out - t_prev))) \
                / np.sqrt(1 - np.cos(t_out)*np.cos(t_prev))
        bmin  = (1/np.sqrt(2)) * (np.sin(0.5*(t_out + t_prev)) + np.sin(0.5*(t_out - t_prev))) \
                / np.sqrt(1 - np.cos(t_out)*np.cos(t_prev))

        U = np.array([
            [aplus,   0.0,    0.0,    amin],
            [amin,    0.0,    0.0,   -aplus],
            [0.0,     bplus,  bmin,   0.0],
            [0.0,     bmin,  -bplus,  0.0],
        ]).T
        return U

    def generate_cloner_circuit(
        self,
        graph: nx.Graph,
        occurances: Dict[str, int],
        qubit_mapping: Dict[str, cirq.Qid],
        n_qubits: int,
    ) -> cirq.Circuit:
        """Build a Cirq circuit that sequentially applies cloning unitaries."""
        circuit = cirq.Circuit()
        
        for i in range(len(occurances)):
            var = f"x{i}"
            M = occurances.get(var, 1)
            if M <= 1:
                continue  # no cloning needed

            # 1→M equatorial cloning angles
            thetas = [np.arccos(np.cos(self.theta) ** (k / M)) for k in range(1, M)]
            
            # original qubit
            last = qubit_mapping[f"x{i}_0"]
            for k, theta_k in enumerate(thetas, start=1):
                theta_prev = thetas[k-2] if k > 1 else theta_k
                # map to the k-th output qubit
                out_label = f"y{i}_{k}"
                target = qubit_mapping[out_label]
                # apply the two-qubit cloning unitary
                matrix = self.cloner_unitary(theta_k, theta_prev)
                gate = cirq.MatrixGate(matrix)
                circuit.append(gate(target, last))
                last = target

        return circuit 