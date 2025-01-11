import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate
from qiskit.circuit import Gate

class SCS2Gate(Gate):
    def __init__(
        self,
        n: int,
    ) -> None:
        self.n = n
        super().__init__("SCS2", 2, [])

    def _define(self):
        qc = QuantumCircuit(2, name='SCS2')

        qc.cx(1, 0)

        theta = 2*np.arccos(np.sqrt(1/self.n))
        cry = RYGate(theta).control(ctrl_state="1")
        qc.append(cry, [0, 1])

        qc.cx(1, 0)

        self.definition = qc

class SCS3Gate(Gate):
    def __init__(
        self,
        l: int,
        n: int,
    ) -> None:
        self.l = l
        self.n = n
        super().__init__("SCS3", 3, [])

    def _define(self):
        qc = QuantumCircuit(3, name='SCS3')

        qc.cx(2, 0)

        theta = 2*np.arccos(np.sqrt(self.l/self.n))
        ccry = RYGate(theta).control(num_ctrl_qubits = 2, ctrl_state="11")
        qc.append(ccry, [0, 1, 2])

        qc.cx(2, 0)

        self.definition = qc

class SCSnkGate(Gate):
    def __init__(
        self,
        n: int,
        k: int,
    ) -> None:
        self.n = n
        self.k = k
        super().__init__(f"SCS{self.n},{self.k}", self.k+1, [])

    def _define(self):
        qc = QuantumCircuit(self.k+1, name=f"SCS{self.n},{self.k}")

        qc.append(SCS2Gate(n=self.n), [0, 1])

        for l in range(2, self.k+1):
            qc.append(SCS3Gate(l=l, n=self.n), [0, l-1, l])

        self.definition = qc

class DickeStatePreparation(QuantumCircuit):
    def __init__(
        self,
        n,
        k
        ):
        self.n = n
        self.k = k
        
        qc = self.circuit()
        super().__init__(*qc.qregs, name=qc.name)
        self.compose(qc.to_gate(), qubits=self.qubits, inplace=True)
    
    def circuit(self):
        n = self.n
        k = self.k
        label = f"{n},{k}"
        qc = QuantumCircuit(n, name=f"$U_{{{label}}}$")
        for i in range(k):
            qc.x(i)
            
        for l in range(n, k, -1):
            qc.append(SCSnkGate(n=l, k=k), range(n-l, n-l+k+1))

        for l in range(k, 1, -1):
            qc.append(SCSnkGate(n=l, k=l-1), range(n-l, n))

        return qc