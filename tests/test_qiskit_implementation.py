"""Test script for Qiskit implementation of DQI circuits.

This script tests both Belief Propagation (BP) and Gauss-Jordan Elimination (GJE)
decoding methods and visualizes the results.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Add the src directory to Python path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root)

# Qiskit imports
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Statevector, partial_trace

# DQI imports
from src.dqi.qiskit.initialization.state_preparation.gates import UnaryAmplitudeEncoding
from src.dqi.qiskit.initialization.calculate_w import get_optimal_w
from src.dqi.qiskit.dicke_state_preparation.gates import UnkGate
from src.dqi.qiskit.decoding.gates import GJEGate
from src.dqi.qiskit.decoding.BPQM.linearcode import LinearCode
from src.dqi.qiskit.decoding.BPQM.decoders import create_init_qc, decode_single_syndrome
from src.dqi.qiskit.decoding.BPQM.cloner import VarNodeCloner
from src.dqi.utils.solver import brute_force_max
from src.dqi.utils.visualize import plot_results_union_plotly
from src.dqi.utils.counts import post_selection_counts, combine_counts


def create_test_matrix() -> Tuple[np.ndarray, np.ndarray]:
    """Create a test parity check matrix H and vector v."""
    H = np.array([
        [1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 1]
    ]).T
    B = H.T
    v = np.ones(B.shape[0])
    return H, B, v


def test_gje_decoding(B: np.ndarray, v: np.ndarray, shots: int = 10000) -> Dict[str, int]:
    """Test GJE decoding method."""
    print("\n" + "="*60)
    print("Testing Gauss-Jordan Elimination (GJE) Decoding")
    print("="*60)
    
    # Get dimensions
    m, n = B.shape
    
    # Calculate optimal weights
    p, r, ell = 2, 1, 2
    w = get_optimal_w(m, ell, p, r)
    print(f"Optimal weights: {w}")
    
    # Create initialization circuit
    init_qregs = QuantumRegister(m, name='k')
    initialize_circuit = QuantumCircuit(init_qregs)
    WUE_Gate = UnaryAmplitudeEncoding(num_bit=m, weights=w)
    initialize_circuit.append(WUE_Gate, range(m))
    
    # Create Dicke state circuit
    dicke_qregs = QuantumRegister(m, name='y')
    dicke_circuit = QuantumCircuit(dicke_qregs)
    max_errors = int(np.nonzero(w)[0][-1]) if np.any(w) else 0
    dicke_circuit.append(UnkGate(m, max_errors), range(m))
    
    # Create registers
    dicke_cregs = ClassicalRegister(m, name='cy')
    syndrome_qregs = QuantumRegister(n, name='syndrome')
    syndrome_cregs = ClassicalRegister(n, name='csolution')
    
    # Build DQI circuit
    dqi_circuit = QuantumCircuit(dicke_qregs, syndrome_qregs, syndrome_cregs, dicke_cregs)
    dqi_circuit.compose(initialize_circuit, inplace=True)
    dqi_circuit.barrier()
    dqi_circuit.compose(dicke_circuit, inplace=True)
    dqi_circuit.barrier()
    
    # Apply phase flips based on v
    for i in range(len(v)):
        if v[i] == 1:
            dqi_circuit.z(i)
    dqi_circuit.barrier()
    
    # Encode constraint matrix B
    for i in range(n):
        for j in range(m):
            if B.T[i][j] == 1:
                dqi_circuit.cx(j, m+i)
    dqi_circuit.barrier()
    # Add GJE decoding
    if n > m:
        decoding_circuit = QuantumCircuit(syndrome_qregs, name="GJE")
        GJE_gate = GJEGate(B.T)
        decoding_circuit.append(GJE_gate, range(n))
        dqi_circuit.append(decoding_circuit, range(m, m+n))
        for i in range(m):
            dqi_circuit.cx(syndrome_qregs[i], dicke_qregs[i])
        dqi_circuit.append(decoding_circuit.inverse(), range(m, m+n))
    else:
        # For n <= m case, we only apply inverse GJE before CNOTs
        decoding_circuit = QuantumCircuit(dicke_qregs[:n], name="GJE")
        GJE_gate = GJEGate(B.T)
        decoding_circuit.append(GJE_gate, range(n))
        dqi_circuit.append(decoding_circuit.inverse(), range(n))
        for i in range(n):
            dqi_circuit.cx(syndrome_qregs[i], dicke_qregs[i])
    
    # Add Hadamard gates
    dqi_circuit.barrier()
    for i in range(n):
        dqi_circuit.h(m+i)
    
    # Add measurements
    dqi_circuit.barrier()
    dqi_circuit.measure(dicke_qregs, dicke_cregs)
    dqi_circuit.measure(syndrome_qregs[::-1], syndrome_cregs)
    
    # Run simulation
    print(f"Running simulation with {shots} shots...")
    simulator = AerSimulator()
    transpiled_circuit = transpile(dqi_circuit, backend=simulator)
    result = simulator.run(transpiled_circuit, shots=shots).result()
    counts = result.get_counts(dqi_circuit)
    
    print(f"Circuit depth: {dqi_circuit.depth()}")
    print(f"Number of qubits: {dqi_circuit.num_qubits}")
    print(f"Number of unique measurement outcomes: {len(counts)}")
    
    return counts


def test_bp_decoding(B: np.ndarray, v: np.ndarray, shots: int = 10000) -> Dict[str, int]:
    """Test Belief Propagation decoding method."""
    print("\n" + "="*60)
    print("Testing Belief Propagation (BP) Decoding")
    print("="*60)
    
    m, n = B.shape
    
    # Calculate optimal weights
    p, r, ell = 2, 1, 2
    w = get_optimal_w(m, ell, p, r)
    print(f"Optimal weights: {w}")
    
    # Create initialization circuit
    init_qregs = QuantumRegister(m, name='k')
    initialize_circuit = QuantumCircuit(init_qregs)
    WUE_Gate = UnaryAmplitudeEncoding(num_bit=m, weights=w)
    initialize_circuit.append(WUE_Gate, range(m))
    
    # Create Dicke state circuit
    dicke_qregs = QuantumRegister(m, name='y')
    dicke_circuit = QuantumCircuit(dicke_qregs)
    max_errors = int(np.nonzero(w)[0][-1]) if np.any(w) else 0
    dicke_circuit.append(UnkGate(m, max_errors), range(m))
    
    # Setup BP decoding
    theta = 0.2 * np.pi
    cloner = VarNodeCloner(theta)
    code = LinearCode(None, B.T)
    syndrome_qc = QuantumCircuit(code.hk)
    
    try:
        decoded_bits, decoded_qubits, qc_decode = decode_single_syndrome(
            syndrome_qc=syndrome_qc,
            code=code,
            prior=0.5,
            theta=theta,
            height=2,
            shots=1024,
            debug=True,
            run_simulation=False
        )
    except Exception as e:
        print(f"BP decoding failed: {str(e)}")
        print("This might be due to the matrix structure or timeout.")
        return {}
    
    # Create registers
    dicke_cregs = ClassicalRegister(m, name='cy')
    syndrome_qregs = QuantumRegister(n, name='syndrome')
    syndrome_cregs = ClassicalRegister(syndrome_qregs.size, name='csolution')
    ancilla_qregs = QuantumRegister(qc_decode.num_qubits - m, name='ancilla')
    
    # Build DQI circuit
    dqi_circuit = QuantumCircuit(dicke_qregs, syndrome_qregs, syndrome_cregs, dicke_cregs, ancilla_qregs)
    dqi_circuit.compose(initialize_circuit, inplace=True)
    dqi_circuit.barrier()
    dqi_circuit.compose(dicke_circuit, inplace=True)
    dqi_circuit.barrier()
    
    # Apply phase flips based on v
    for i in range(len(v)):
        if v[i] == 1:
            dqi_circuit.z(i)
    dqi_circuit.barrier()
    
    # Encode constraint matrix B
    for i in range(n):
        for j in range(m):
            if B.T[i][j] == 1:
                dqi_circuit.cx(j, m+i)
    dqi_circuit.barrier()
    
    # Add BP decoding
    dqi_circuit.compose(
        qc_decode,
        qubits=list(range(dicke_qregs.size, qc_decode.num_qubits)) + list(range(dicke_qregs.size)),
        inplace=True
    )
    
    # Add Hadamard gates
    for i in range(n):
        dqi_circuit.h(m+i)
    
    # Add measurements
    dqi_circuit.barrier()
    dqi_circuit.measure(dicke_qregs, dicke_cregs)
    dqi_circuit.measure(syndrome_qregs[::-1], syndrome_cregs)
    
    # Run simulation
    print(f"Running simulation with {shots} shots...")
    simulator = AerSimulator()
    transpiled_circuit = transpile(dqi_circuit, backend=simulator)
    result = simulator.run(transpiled_circuit, shots=shots).result()
    counts = result.get_counts(dqi_circuit)
    
    print(f"Circuit depth: {dqi_circuit.depth()}")
    print(f"Number of qubits: {dqi_circuit.num_qubits}")
    print(f"Number of unique measurement outcomes: {len(counts)}")
    
    return counts


def plot_comparison(brute_force_results: List[Tuple[str, int]], gje_counts: Dict, bp_counts: Dict):
    """Plot comparison of brute force, GJE, and BP results (post-selected only, with dual y-axes)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Qiskit DQI Implementation Test Results (Post-selected)", fontsize=16)

    def prepare_data(counts):
        if not counts:
            return [], []
        solutions = {}
        for bitstring, count in counts.items():
            solution = bitstring.split()[0]  # No reversal
            solutions[solution] = solutions.get(solution, 0) + count
        sorted_items = sorted(solutions.items(), key=lambda x: int(x[0], 2))
        x_vals = [int(k, 2) for k, v in sorted_items]
        y_vals = [v for k, v in sorted_items]
        total = sum(y_vals)
        if total > 0:
            y_vals = [y / total for y in y_vals]
        return x_vals, y_vals

    bf_dict = {int(bitstring, 2): score for bitstring, score in brute_force_results}
    bf_sorted = sorted(bf_dict.items())
    bf_x = [x for x, _ in bf_sorted]
    bf_y = [y for _, y in bf_sorted]

    gje_x, gje_y = prepare_data(gje_counts) if gje_counts else ([], [])
    bp_x, bp_y = prepare_data(bp_counts) if bp_counts else ([], [])

    # Brute Force vs GJE
    ax1 = axes[0]
    ax2 = ax1.twinx()
    ax1.bar(bf_x, bf_y, alpha=0.7, color='blue', label='Brute Force')
    if gje_x:
        ax2.plot(gje_x, gje_y, 'g-o', label='GJE', markersize=6)
        max_prob = max(abs(y) for y in gje_y) if gje_y else 0.1
        ax2.set_ylim(-max_prob, max_prob)
    else:
        max_prob = 0.1
        ax2.set_ylim(-max_prob, max_prob)
    ax1.set_title("Brute Force vs GJE (Post-selected)")
    ax1.set_xlabel("Solution (decimal)")
    ax1.set_ylabel("Objective Value", color='blue')
    ax2.set_ylabel("Probability", color='green')
    ax1.legend(loc='upper left')
    if gje_x:
        ax2.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Brute Force vs BP
    ax1 = axes[1]
    ax2 = ax1.twinx()
    ax1.bar(bf_x, bf_y, alpha=0.7, color='blue', label='Brute Force')
    if bp_x:
        ax2.plot(bp_x, bp_y, 'r-s', label='BP', markersize=6)
        max_prob = max(abs(y) for y in bp_y) if bp_y else 0.1
        ax2.set_ylim(-max_prob, max_prob)
    else:
        max_prob = 0.1
        ax2.set_ylim(-max_prob, max_prob)
    ax1.set_title("Brute Force vs BP (Post-selected)")
    ax1.set_xlabel("Solution (decimal)")
    ax1.set_ylabel("Objective Value", color='blue')
    ax2.set_ylabel("Probability", color='red')
    ax1.legend(loc='upper left')
    if bp_x:
        ax2.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("qiskit_test_results.png", dpi=150)
    plt.show()


def main():
    """Main test function."""
    print("="*60)
    print("DQI Qiskit Implementation Test")
    print("="*60)
    
    # Create test matrix
    H, B, v = create_test_matrix()
    print(f"Test matrix H shape: {H.shape}")
    print(f"Matrix B (H^T) shape: {B.shape}")
    print(f"Vector v: {v}")
    
    # Compute brute force results
    print("\nComputing brute force results...")
    brute_force_list = brute_force_max(B, v)
    brute_force_results = brute_force_list  # Keep as list of tuples
    print(f"Number of solutions: {len(brute_force_results)}")
    print(f"Max objective value: {max(score for _, score in brute_force_results)}")
    
    # Test GJE decoding
    gje_counts_raw = test_gje_decoding(B, v, shots=10000)
    gje_counts = post_selection_counts(gje_counts_raw)
    
    # Test BP decoding
    bp_counts_raw = test_bp_decoding(B, v, shots=10000)
    bp_counts = post_selection_counts(bp_counts_raw)
    
    # Plot results (post-selected only)
    print("\nGenerating comparison plots...")
    plot_comparison(brute_force_results, gje_counts, bp_counts)
    
    print("\nTest completed! Results saved to 'qiskit_test_results.png'")


if __name__ == "__main__":
    main() 