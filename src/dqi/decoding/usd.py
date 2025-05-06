import math
import numpy as np
import logging
from typing import List, Tuple, Optional
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator

def compute_usd_povm(
    statevecs: List[np.ndarray], priors: List[float]
) -> List[Operator]:
    """
    Compute unambiguous discrimination POVM elements for two pure qubit states.

    Args:
        statevecs: List of two numpy arrays each of length 2.
        priors: List of two prior probabilities.

    Returns:
        List of three Operators [Pi1, Pi2, Pifail].
    """
    if len(statevecs) != 2 or len(priors) != 2:
        raise ValueError("Need exactly two states and two priors")

    psi1, psi2 = statevecs
    p1, p2 = priors
    ortho1 = np.array([-psi1[1], psi1[0]])
    ortho2 = np.array([-psi2[1], psi2[0]])
    c = abs(np.vdot(psi1, psi2))
    s = abs(np.vdot(psi1, ortho2))

    left = c**2 / (1 + c**2)
    right = 1 / (1 + c**2)
    if not (left <= p1 <= right):
        raise ValueError("Priors out of valid range for USD")

    q1 = math.sqrt(p2 / p1) * c
    q2 = math.sqrt(p1 / p2) * c
    Pi1 = (1 - q1) / s**2 * np.outer(ortho2, np.conj(ortho2))
    Pi2 = (1 - q2) / s**2 * np.outer(ortho1, np.conj(ortho1))
    Pif = np.eye(2) - Pi1 - Pi2

    return [Operator(Pi1), Operator(Pi2), Operator(Pif)]


def build_usd_unitaries(
    povm_elems: List[Operator]
) -> List[UnitaryGate]:
    """
    Build 1-qubit UnitaryGates that map each POVM element's principal eigenvector to |0>.
    """
    gates: List[UnitaryGate] = []
    for Pi in povm_elems:
        vals, vecs = np.linalg.eigh(Pi.data)
        v = vecs[:, np.argmax(vals)]
        w = np.array([-v[1].conj(), v[0].conj()])
        U = np.vstack([v.conj(), w.conj()])
        gates.append(UnitaryGate(U))
    return gates


def apply_usd_once(
    qc: QuantumCircuit,
    wires: List[int],
    p: float,
    simulator: Optional[AerSimulator] = None
) -> Tuple[np.ndarray, QuantumCircuit]:
    """
    Perform one round of USD discrimination on `qc` at given `wires` with overlap p.

    Returns:
        recovered mask (1D numpy array) and the final measured circuit.
    """
    sim = simulator or AerSimulator()
    # prepare POVM and unitaries
    psi0 = Statevector([np.sqrt(1-p), np.sqrt(p)])
    psi1 = Statevector([np.sqrt(p), np.sqrt(1-p)])
    povm = compute_usd_povm([psi0.data, psi1.data], [0.5, 0.5])
    U1, U2, _ = build_usd_unitaries(povm)

    n = len(wires)
    recovered = np.full(n, -1, dtype=int)

    # test Pi1
    qc1 = qc.copy()
    for i, w in enumerate(wires):
        qc1.append(U1, [w])
    qc1.measure(wires, wires)
    counts1 = sim.run(transpile(qc1, sim), shots=1).result().get_counts()
    bits1 = next(iter(counts1)).split()[::-1][0]
    for i, b in enumerate(bits1):
        if b == '0':
            recovered[i] = 0

    # test Pi2 on undecided
    qc2 = qc.copy()
    for i, w in enumerate(wires):
        qc2.append(U2, [w])
    qc2.measure(wires, wires)
    counts2 = sim.run(transpile(qc2, sim), shots=1).result().get_counts()
    bits2 = next(iter(counts2)).split()[::-1][0]
    for i, b in enumerate(bits2):
        if recovered[i] < 0 and b == '0':
            recovered[i] = 1

    return recovered, qc2


def usd_reduction(
    qc: QuantumCircuit,
    B: np.ndarray,
    wires: List[int],
    p: float = 0.0,
    simulator: Optional[AerSimulator] = None,
    max_attempts: int = 100
) -> np.ndarray:
    """
    Repeatedly apply USD until at least one conclusive bit per constraint or max_attempts.

    Returns:
        Reduced matrix columns corresponding to conclusive indices.
    """
    m, n = B.shape
    for attempt in range(1, max_attempts + 1):
        recovered, _ = apply_usd_once(qc, wires, p, simulator)
        idx = np.where(recovered >= 0)[0]
        if len(idx) >= m:
            return B[:, idx]
        print(f"USD attempt {attempt} inconclusive: {recovered}")
    print(f"Exceeded {max_attempts} USD attempts; returning original B")
    return B
