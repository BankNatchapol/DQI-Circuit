import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import RYGate

class SCS2Gate(Gate):
    """
    Implements the SCS2 gate for a 2-qubit system.
    
    The gate is defined by the following steps:
      1. Apply a CNOT with qubit 1 as control and qubit 0 as target.
      2. Apply a controlled RY rotation on qubits 0 and 1 with rotation angle
         theta = 2 * arccos(1/sqrt(n)), where n is provided at construction.
      3. Apply a second CNOT with qubit 1 as control and qubit 0 as target.
    """
    def __init__(self, n: int) -> None:
        self.n: int = n
        super().__init__("SCS2", 2, [])

    def _define(self) -> None:
        qc = QuantumCircuit(2, name='SCS2')
        qc.cx(1, 0)
        theta: float = 2 * np.arccos(1 / np.sqrt(self.n))
        cry = RYGate(theta).control(ctrl_state="1")
        qc.append(cry, [0, 1])
        qc.cx(1, 0)
        self.definition = qc

class SCS3Gate(Gate):
    """
    Implements the SCS3 gate for a 3-qubit system.
    
    The gate is defined by:
      1. Applying a CNOT from qubit 2 (control) to qubit 0 (target).
      2. Applying a double-controlled RY rotation on qubits 0, 1, and 2 with
         rotation angle theta = 2 * arccos(sqrt(l/n)), where l and n are provided.
      3. Applying another CNOT from qubit 2 to qubit 0.
    """
    def __init__(self, l: int, n: int) -> None:
        self.l: int = l
        self.n: int = n
        super().__init__("SCS3", 3, [])

    def _define(self) -> None:
        qc = QuantumCircuit(3, name='SCS3')
        qc.cx(2, 0)
        theta: float = 2 * np.arccos(np.sqrt(self.l / self.n))
        ccry = RYGate(theta).control(num_ctrl_qubits=2, ctrl_state="11")
        qc.append(ccry, [0, 1, 2])
        qc.cx(2, 0)
        self.definition = qc

class SCSnkGate(Gate):
    """
    Constructs a composite SCS gate acting on k+1 qubits.
    
    The gate is built by composing:
      - An SCS2Gate on the first two qubits.
      - For each l in 2 to k, an SCS3Gate on qubits [0, l-1, l].
    
    Attributes:
      n (int): Parameter used in the rotation angles.
      k (int): Determines the number of qubits (gate acts on k+1 qubits).
    """
    def __init__(self, n: int, k: int) -> None:
        self.n: int = n
        self.k: int = k
        super().__init__(f"SCS{n},{k}", self.k + 1, [])

    def _define(self) -> None:
        qc = QuantumCircuit(self.k + 1, name=f"SCS{self.n},{self.k}")
        qc.append(SCS2Gate(n=self.n), [0, 1])
        for l in range(2, self.k + 1):
            qc.append(SCS3Gate(l=l, n=self.n), [0, l - 1, l])
        self.definition = qc

class UnkStatePreparation(QuantumCircuit):
    """
    Constructs a state-preparation circuit (U) for unknown state preparation.
    
    The circuit is built by composing several SCSnkGates in two stages.
    The first stage applies gates on overlapping qubit subsets in descending order,
    and the second stage applies additional gates to refine the state.
    
    Attributes:
      n (int): Number of qubits in the circuit.
      k (int): Parameter that affects the gate construction.
    """
    def __init__(self, n: int, k: int) -> None:
        self.n: int = n
        self.k: int = k
        qc: QuantumCircuit = self.circuit()
        super().__init__(*qc.qregs, name=qc.name)
        self.compose(qc.to_gate(), qubits=self.qubits, inplace=True)

    def circuit(self) -> QuantumCircuit:
        n: int = self.n
        k: int = self.k
        label: str = f"{n},{k}"
        qc: QuantumCircuit = QuantumCircuit(n, name=f"$U_{{{label}}}$")
        # First stage: Apply SCSnkGates on overlapping subsets (descending order)
        for l in range(n, k, -1):
            qc.append(SCSnkGate(n=l, k=k), list(range(n - l, n - l + k + 1)))
        # Second stage: Refine the state with additional SCSnkGates
        for l in range(k, 1, -1):
            qc.append(SCSnkGate(n=l, k=l - 1), list(range(n - l, n)))
        return qc

class DickeStatePreparation(QuantumCircuit):
    """
    Constructs a Dicke state preparation circuit (D) for a system with n qubits and k excitations.
    
    The circuit starts by initializing the first k qubits in the |1> state and then applies a series
    of SCSnkGates to entangle the qubits into a Dicke state.
    
    Attributes:
      n (int): Number of qubits.
      k (int): Number of excitations (qubits initially set to |1>).
    """
    def __init__(self, n: int, k: int) -> None:
        self.n: int = n
        self.k: int = k
        qc: QuantumCircuit = self.circuit()
        super().__init__(*qc.qregs, name=qc.name)
        self.compose(qc.to_gate(), qubits=self.qubits, inplace=True)

    def circuit(self) -> QuantumCircuit:
        n: int = self.n
        k: int = self.k
        label: str = f"{n},{k}"
        qc: QuantumCircuit = QuantumCircuit(n, name=f"$D_{{{label}}}$")
        # Initialize the first k qubits to |1>
        for i in range(k):
            qc.x(i)
        # First stage: Apply SCSnkGates on overlapping subsets (descending order)
        for l in range(n, k, -1):
            qc.append(SCSnkGate(n=l, k=k), list(range(n - l, n - l + k + 1)))
        # Second stage: Refine the state with additional SCSnkGates
        for l in range(k, 1, -1):
            qc.append(SCSnkGate(n=l, k=l - 1), list(range(n - l, n)))
        return qc
