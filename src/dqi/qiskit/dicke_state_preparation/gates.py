import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import RYGate


def _apply_stages(qc: QuantumCircuit, n: int, k: int):
    """
    Apply two-stage SCSnk gates for U_{n,k} and Dicke preparation:
      1) For l from n down to k+1, apply SCSnk(n=l, k=k) on qubits [n-l : n-l+k+1]
      2) For l from k down to 2, apply SCSnk(n=l, k=l-1) on qubits [n-l : n]
    """
    # Stage 1: overlapping gates descending
    for l in range(n, k, -1):
        start = n - l
        qc.append(SCSnkGate(n=l, k=k), list(range(start, start + k + 1)))
    # Stage 2: refinement gates
    for l in range(k, 1, -1):
        start = n - l
        qc.append(SCSnkGate(n=l, k=l - 1), list(range(start, n)))


class SCS2Gate(Gate):
    """
    SCS gate for 2 qubits: CNOT-CRY-CNOT with theta=2*arccos(1/sqrt(n)).
    """
    def __init__(self, n: int) -> None:
        super().__init__("SCS2", 2, [])
        self.n = n

    def _define(self) -> None:
        qc = QuantumCircuit(2, name="SCS2")
        qc.cx(1, 0)
        theta = 2 * np.arccos(1 / np.sqrt(self.n))
        cry = RYGate(theta).control(1)
        qc.append(cry, [0, 1])
        qc.cx(1, 0)
        self.definition = qc


class SCS3Gate(Gate):
    """
    SCS gate for 3 qubits: CNOT-CCRY-CNOT with theta=2*arccos(sqrt(l/n)).
    """
    def __init__(self, l: int, n: int) -> None:
        super().__init__("SCS3", 3, [])
        self.l = l
        self.n = n

    def _define(self) -> None:
        qc = QuantumCircuit(3, name="SCS3")
        qc.cx(2, 0)
        theta = 2 * np.arccos(np.sqrt(self.l / self.n))
        ccry = RYGate(theta).control(2)
        qc.append(ccry, [0, 1, 2])
        qc.cx(2, 0)
        self.definition = qc


class SCSnkGate(Gate):
    """
    Composite SCS gate on k+1 qubits: one SCS2 then SCS3 for each extra qubit.
    """
    def __init__(self, n: int, k: int) -> None:
        super().__init__(f"SCS{n},{k}", k + 1, [])
        self.n = n
        self.k = k

    def _define(self) -> None:
        qc = QuantumCircuit(self.k + 1, name=self.name)
        qc.append(SCS2Gate(self.n), [0, 1])
        for l in range(2, self.k + 1):
            qc.append(SCS3Gate(l, self.n), [0, l - 1, l])
        self.definition = qc


class UnkGate(Gate):
    """
    Constructs the U_{n,k} state preparation gate by composing two-stage SCSnk sequence.
    """
    def __init__(self, n: int, k: int) -> None:
        super().__init__(f"U$_{{{n},{k}}}$", n, [])
        self.n = n
        self.k = k

    def _define(self) -> None:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        _apply_stages(qc, self.n, self.k)
        self.definition = qc


class DickeStatePreparation(Gate):
    """
    Constructs the D_{n,k} Dicke state gate: initialize k qubits then apply U_{n,k} stages.
    """
    def __init__(self, n: int, k: int) -> None:
        super().__init__(f"D_{{{n},{k}}}", n, [])
        self.n = n
        self.k = k

    def _define(self) -> None:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        for i in range(self.k):
            qc.x(i)
        _apply_stages(qc, self.n, self.k)
        self.definition = qc
