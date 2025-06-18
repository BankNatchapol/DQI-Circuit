"""Cirq implementation of Unary Amplitude Encoding gate."""

import numpy as np
import cirq
from typing import Union, Sequence, List, Tuple


class UnaryAmplitudeEncoding(cirq.Gate):
    """
    Unary Amplitude Encoding (UAE) gate prepares a weighted unary state by applying
    a series of rotations. For each weight index l, the rotation angle is:

        beta_l = 2 * arccos(min(sqrt(weights[l]^2 / (1 - sum_{j<l} weights[j]^2)), 1))

    The first qubit is rotated with RY, and subsequent qubits use controlled RY.

    Args:
        num_bit: Number of qubits.
        weights: A normalized array or sequence of floats.
    """
    
    def __init__(self, num_bit: int, weights: Union[np.ndarray, Sequence[float]]) -> None:
        super().__init__()
        self.num_bit = num_bit
        self.weights = np.array(weights, dtype=float)
        self._name = "UAE"
    
    def _num_qubits_(self) -> int:
        return self.num_bit
    
    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(
            wire_symbols=[f"UAE_{i}" for i in range(self.num_bit)]
        )
    
    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        """Decompose the UAE gate into basic gates."""
        # Vectorized beta computation
        w2 = self.weights ** 2
        cum = np.concatenate(([0.0], np.cumsum(w2[:-1])))  # sum_{j<l} weights[j]^2
        denom = 1.0 - cum
        # Avoid division by zero, clip ratios
        ratio = np.where(denom > 0, w2 / denom, 0.0)
        ratio = np.clip(ratio, 0.0, 1.0)
        betas = 2.0 * np.arccos(np.sqrt(ratio))
        
        operations = []
        
        # Apply first RY if nonzero
        if betas[0] != 0.0 and not np.isnan(betas[0]):
            operations.append(cirq.ry(betas[0]).on(qubits[0]))
        
        # Controlled rotations
        for i in range(1, self.num_bit):
            if i < betas.size:
                b = betas[i]
                if b != 0.0 and not np.isnan(b):
                    # Use ControlledGate for controlled RY
                    operations.append(
                        cirq.ControlledGate(cirq.ry(b), num_controls=1).on(qubits[i-1], qubits[i])
                    )
        
        return operations
    
    def __repr__(self) -> str:
        return f"UnaryAmplitudeEncoding(num_bit={self.num_bit}, weights={self.weights})" 