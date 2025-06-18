import numpy as np
from qiskit.circuit import Gate, QuantumCircuit
from typing import Union, Sequence

class UnaryAmplitudeEncoding(Gate):
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
        super().__init__("UAE", num_bit, [])
        self.num_bit = num_bit
        self.weights = np.array(weights, dtype=float)

    def _define(self) -> None:
        qc = QuantumCircuit(self.num_bit, name="UAE")
        # Vectorized beta computation
        w2 = self.weights ** 2
        cum = np.concatenate(([0.0], np.cumsum(w2[:-1])))  # sum_{j<l} weights[j]^2
        denom = 1.0 - cum
        # Avoid division by zero, clip ratios
        ratio = np.where(denom > 0, w2 / denom, 0.0)
        ratio = np.clip(ratio, 0.0, 1.0)
        betas = 2.0 * np.arccos(np.sqrt(ratio))

        # Apply first RY if nonzero
        if betas[0] != 0.0 and not np.isnan(betas[0]):
            qc.ry(betas[0], 0)

        # Controlled rotations
        for i in range(1, self.num_bit):
            if i < betas.size:
                b = betas[i]
                if b != 0.0 and not np.isnan(b):
                    qc.cry(b, i - 1, i)

        self.definition = qc
