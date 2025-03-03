from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from itertools import product
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
    def __init__(self, num_bit, weights):
        """
        Create a weighted unary state preparation gate.

        Args:
            num_bit (int): Number of qubits.
            weights (list or np.ndarray): A normalized list of weights.
        """
        # Optionally: Add a check to ensure weights are normalized or that len(weights) is acceptable.
        super().__init__("WeightedUnary", num_bit, [])
        self.num_bit = num_bit
        self.weights = weights

    def _define(self):
        qc = QuantumCircuit(self.num_bit, name="WeightedUnary")
        # Compute the rotation angles.
        # Note: For each weight l, we compute beta_l = 2 * arccos( min( sqrt(weights[l]^2 / (1 - sum(weights[:l]^2)), 1) ).
        betas = np.array([
            min(np.sqrt(self.weights[l]**2 / (1 - np.sum(np.array(self.weights[:l])**2))), 1)
            for l in range(len(self.weights))
        ])
        betas = 2 * np.arccos(betas)

        # Apply the first rotation on qubit 0.
        qc.ry(betas[0], 0)
        # Apply controlled Ry rotations on subsequent qubits.
        for i in range(1, self.num_bit):
            if i >= len(betas):
                continue
            if np.isnan(betas[i]):
                continue
            qc.cry(betas[i], i - 1, i)
        self.definition = qc
