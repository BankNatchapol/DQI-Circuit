from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from typing import Union, Sequence
import numpy as np

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


class WeightedUnaryEncoding(Gate):
    """
    Weighted Unary Encoding gate for state preparation.

    This gate prepares a weighted unary state by applying a series of rotations.
    For each weight index l, the rotation angle is computed as:

        beta_l = 2 * arccos( min( sqrt(weights[l]^2 / (1 - sum_{j<l} weights[j]^2)), 1) )

    The first qubit is rotated with an RY gate and subsequent qubits are rotated
    with controlled RY (CRY) gates.

    Args:
        num_bit (int): Number of qubits.
        weights (Union[np.ndarray, Sequence[float]]): A normalized list of weights.
    """
    def __init__(self, num_bit: int, weights: Union[np.ndarray, Sequence[float]]) -> None:
        super().__init__("Weighted_Unary", num_bit, [])
        self.num_bit: int = num_bit
        # Ensure weights are stored as a numpy array of floats.
        self.weights: np.ndarray = np.array(weights, dtype=float)

    def _define(self) -> None:
        """
        Define the internal structure of the Weighted Unary Encoding gate.
        """
        qc = QuantumCircuit(self.num_bit, name="Weighted_Unary")
        
        # Compute rotation angles for each weight.
        betas = []
        for l in range(len(self.weights)):
            # Denom = 1 - sum_{j=0}^{l-1} (weights[j]^2)
            denom = 1 - np.sum(self.weights[:l] ** 2)
            # To avoid division by zero, set ratio to 0 when denom is zero.
            ratio = (self.weights[l] ** 2) / denom if denom > 0 else 0.0
            # Ensure the argument for arccos is at most 1.
            value = min(np.sqrt(ratio), 1.0)
            beta = 2 * np.arccos(value)
            betas.append(beta)
        betas = np.array(betas)

        # Apply the first rotation on qubit 0.
        qc.ry(betas[0], 0)
        # Apply controlled RY rotations on subsequent qubits.
        for i in range(1, self.num_bit):
            if i >= len(betas):
                continue
            if np.isnan(betas[i]) or betas[i] == 0.0:
                continue
            qc.cry(betas[i], i - 1, i)
        
        self.definition = qc
