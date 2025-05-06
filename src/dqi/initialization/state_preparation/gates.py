import numpy as np
from qiskit.circuit import Gate, QuantumCircuit
from typing import Union, Sequence

# class GGate(Gate):
#     def __init__(self, dagger=False):
#         super().__init__('G' + ('\u2020' if dagger else ''), 1, [])
#         self.dagger = dagger

#     def _define(self):
#         qc = QuantumCircuit(1, name='G' + ('\u2020' if self.dagger else ''))
#         if self.dagger:
#             qc.s(0)
#             qc.h(0)
#             qc.t(0)
#             qc.h(0)
#             qc.sdg(0)
#         else:
#             qc.sdg(0)
#             qc.h(0)
#             qc.t(0)
#             qc.h(0)
#             qc.s(0)
#         self.definition = qc

# class PhaseIncorrectToffoliGate(Gate):
#     def __init__(self):
#         super().__init__('Phase_Incorrect_Toffoli', 3, [])

#     def _define(self):
#         qc = QuantumCircuit(3, name='Phase_Incorrect_Toffoli')
#         # Decomposition using G-Gates
#         qc.append(GGate(dagger=True), [2])
#         qc.cx(1, 2)
#         qc.append(GGate(dagger=True), [2])
#         qc.cx(0, 2)
#         qc.append(GGate(dagger=False), [2])
#         qc.cx(1, 2)
#         qc.append(GGate(dagger=False), [2])
#         self.definition = qc

# class PhaseIncorrectCSwapGate(Gate):
#     def __init__(self):
#         super().__init__('Phase_Incorrect_Toffoli', 3, [])

#     def _define(self):
#         qc = QuantumCircuit(3, name='Phase_Incorrect_Toffoli')
#         # Decomposition using G-Gates
#         qc.cx(2, 1)
#         qc.append(GGate(dagger=True), [2])
#         qc.cx(1, 2)
#         qc.append(GGate(dagger=True), [2])
#         qc.cx(0, 2)
#         qc.append(GGate(dagger=False), [2])
#         qc.cx(1, 2)
#         qc.append(GGate(dagger=False), [2])
#         qc.cx(2, 1)
#         self.definition = qc


# def binary_combinations(n):
#     # Generate all combinations of binary of length n
#     return [''.join(map(str, bits)) for bits in product([0, 1], repeat=n)]

# class SelectGate(Gate):
#     def __init__(
#         self,
#         num_ctrl_qubits: int,
#         num_target_qubits: int,
#     ) -> None:
        
#         self.num_ctrl_qubits = num_ctrl_qubits
#         self.num_target_qubits = num_target_qubits

#         # initialize the circuit object
#         num_qubits = num_ctrl_qubits + num_target_qubits
#         self.num_qubits = num_qubits
#         super().__init__(num_qubits, name="Select")

#     def _define(self):
#         combinations = binary_combinations(self.num_ctrl_qubits)
#         qc = QuantumCircuit(self.num_qubits, name='Select')
#         # Decomposition using G-Gates
#         for c in combinations:
#             qc.mcmt([], )
#         self.definition = qc

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
