"""Test script for Cirq implementation of DQI circuits.

This script tests the Gauss-Jordan Elimination (GJE) decoding method
and visualizes the results. BP is not yet implemented for Cirq.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import cirq

# Add the src directory to Python path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root)

# DQI imports
from src.dqi.cirq.initialization.state_preparation.gates import UnaryAmplitudeEncoding
from src.dqi.cirq.initialization.calculate_w import get_optimal_w
from src.dqi.cirq.dicke_state_preparation.gates import UnkGate
from src.dqi.cirq.decoding.gates import GJEGate
from src.dqi.utils.solver import brute_force_max


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


def test_gje_decoding_cirq(B: np.ndarray, v: np.ndarray, shots: int = 10000) -> Dict[str, int]:
    """Test GJE decoding method using Cirq."""
    print("\n" + "="*60)
    print("Testing Gauss-Jordan Elimination (GJE) Decoding with Cirq")
    print("="*60)
    
    # Get dimensions
    m, n = B.shape

    # Calculate optimal weights for unary amplitude encoding
    p, r, ell = 2, 1, 2
    m = B.shape[0]
    w = get_optimal_w(m, ell, p, r)
    print(f"Optimal weights: {w}")
    
    
    # Create qubits
    dicke_qubits = [cirq.NamedQubit(f'y_{i}') for i in range(m)]
    syndrome_qubits = [cirq.NamedQubit(f'syndrome_{i}') for i in range(n)]
    
    # Initialize the circuit
    circuit = cirq.Circuit()
    
    # Add Unary Amplitude Encoding
    uae_gate = UnaryAmplitudeEncoding(num_bit=m, weights=w)
    circuit.append(uae_gate(*dicke_qubits))
    
    # Add barrier (using moment separation)
    circuit.append(cirq.Moment())
    
    # Add Dicke state preparation
    max_errors = int(np.nonzero(w)[0][-1]) if np.any(w) else 0
    unk_gate = UnkGate(m, max_errors)
    circuit.append(unk_gate(*dicke_qubits))
    print(f"Max errors for Dicke state: {max_errors}")
    
    # Add barrier
    circuit.append(cirq.Moment())
    
    # Apply phase flips based on v
    for i in range(len(v)):
        if v[i] == 1:
            circuit.append(cirq.Z(dicke_qubits[i]))
    
    # Add barrier
    circuit.append(cirq.Moment())
    
    # Encode constraint matrix B
    cx_count = 0
    for i in range(n):
        for j in range(m):
            if B[i][j] == 1:
                circuit.append(cirq.CNOT(dicke_qubits[j], syndrome_qubits[i]))
                cx_count += 1
    print(f"Added {cx_count} CNOT gates for constraints")
    
    # Add barrier
    circuit.append(cirq.Moment())
    
    # Add GJE decoding
    if n > m:
        gje_gate = GJEGate(B)
        circuit.append(gje_gate(*syndrome_qubits))
        for i in range(m):
            circuit.append(cirq.CNOT(syndrome_qubits[i], dicke_qubits[i]))
        # Apply inverse
        circuit.append(gje_gate(*syndrome_qubits))
    else:
        gje_gate = GJEGate(B)
        # Apply inverse first
        inverse_ops = list(gje_gate._decompose_(dicke_qubits[:n]))
        for op in reversed(inverse_ops):
            if isinstance(op.gate, type(cirq.SWAP)):
                circuit.append(op)  # SWAP is its own inverse
            elif isinstance(op.gate, type(cirq.CNOT)):
                circuit.append(op)  # CNOT is its own inverse
        for i in range(n):
            circuit.append(cirq.CNOT(syndrome_qubits[i], dicke_qubits[i]))
    
    # Add Hadamard gates
    circuit.append(cirq.Moment())
    for qubit in syndrome_qubits:
        circuit.append(cirq.H(qubit))
    
    # Add barrier
    circuit.append(cirq.Moment())
    
    # Add measurements
    circuit.append([cirq.measure(q, key=f'c{q.name}') for q in dicke_qubits])
    circuit.append([cirq.measure(q, key=f'c{q.name}') for q in syndrome_qubits])
    
    # Print circuit statistics
    print(f"Circuit depth: {len(circuit)}")
    print(f"Number of qubits: {len(circuit.all_qubits())}")
    
    # Run simulation
    print(f"Running simulation with {shots} shots...")
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=shots)
    
    # Process results
    counts = {}
    # Get the correct measurement keys based on how they were set up in the circuit
    dicke_keys = [f'cy_{i}' for i in range(m)]
    syndrome_keys = [f'csyndrome_{i}' for i in range(n)]
    
    # Check if keys exist in measurements
    available_keys = list(result.measurements.keys())
    print(f"Available measurement keys: {available_keys}")
    
    # Extract measurements using the correct Cirq API
    dicke_results = np.zeros((shots, m), dtype=int)
    syndrome_results = np.zeros((shots, n), dtype=int)
    
    # Fill dicke results
    for i in range(m):
        key = f'cy_{i}'
        if key in available_keys:
            dicke_results[:, i] = result.measurements[key].flatten()
    
    # Fill syndrome results
    for i in range(n):
        key = f'csyndrome_{i}'
        if key in available_keys:
            syndrome_results[:, i] = result.measurements[key].flatten()
    
    for i in range(shots):
        # Combine dicke and syndrome measurements
        dicke_bits = ''.join(str(dicke_results[i, j]) for j in range(m))
        syndrome_bits = ''.join(str(syndrome_results[i, j]) for j in range(n))
        # Reverse syndrome bits to match Qiskit convention
        syndrome_bits = syndrome_bits[::-1]
        bitstring = f"{syndrome_bits} {dicke_bits}"
        counts[bitstring] = counts.get(bitstring, 0) + 1
    
    print(f"Number of unique measurement outcomes: {len(counts)}")
    
    return counts


def plot_cirq_results(brute_force_results: Dict, gje_counts: Dict):
    """Plot comparison of brute force and GJE results for Cirq."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Cirq DQI Implementation Test Results", fontsize=16)
    
    # Helper function to prepare data for plotting
    def prepare_data(counts):
        if not counts:
            return [], []
        # Extract solution part (last n bits)
        solutions = {}
        for bitstring, count in counts.items():
            solution = bitstring.split()[0]  # Get solution part
            solutions[solution] = solutions.get(solution, 0) + count
        
        # Sort by solution value
        sorted_items = sorted(solutions.items(), key=lambda x: int(x[0], 2))
        x_vals = [int(k, 2) for k, v in sorted_items]
        y_vals = [v for k, v in sorted_items]
        
        # Normalize
        total = sum(y_vals)
        if total > 0:
            y_vals = [y / total for y in y_vals]
        
        return x_vals, y_vals
    
    # Plot brute force results
    # Sort by solution value to ensure proper ordering
    sorted_bf_items = sorted(brute_force_results.items(), key=lambda x: x[0])
    bf_x = [item[0] for item in sorted_bf_items]
    bf_y = [item[1] for item in sorted_bf_items]
    axes[0].bar(bf_x, bf_y, alpha=0.7, color='blue')
    axes[0].set_title("Brute Force Objective Values")
    axes[0].set_xlabel("Solution (decimal)")
    axes[0].set_ylabel("Objective Value")
    axes[0].grid(True, alpha=0.3)
    
    # Plot GJE results
    if gje_counts:
        gje_x, gje_y = prepare_data(gje_counts)
        axes[1].bar(gje_x, gje_y, alpha=0.7, color='green')
        axes[1].set_title("Cirq GJE Decoding Results")
        axes[1].set_xlabel("Solution (decimal)")
        axes[1].set_ylabel("Probability")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "GJE Test Failed", ha='center', va='center')
    
    # Combined comparison
    axes[2].bar(bf_x, np.array(bf_y) / max(bf_y), alpha=0.5, label='Brute Force (normalized)', color='blue')
    if gje_counts:
        gje_x, gje_y = prepare_data(gje_counts)
        axes[2].plot(gje_x, gje_y, 'g-o', label='Cirq GJE', markersize=6)
    
    axes[2].set_title("Combined Comparison")
    axes[2].set_xlabel("Solution (decimal)")
    axes[2].set_ylabel("Normalized Value/Probability")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("cirq_test_results.png", dpi=150)
    plt.show()


def main():
    """Main test function."""
    print("="*60)
    print("DQI Cirq Implementation Test")
    print("="*60)
    
    # Create test matrix
    H, B, v = create_test_matrix()
    print(f"Test matrix H shape: {H.shape}")
    print(f"Matrix B (H^T) shape: {B.shape}")
    print(f"Vector v: {v}")
    
    # Compute brute force results
    print("\nComputing brute force results...")
    brute_force_list = brute_force_max(B, v)
    print(brute_force_list)
    # Convert to dictionary with integer keys
    brute_force_results = {int(bitstring, 2): score for bitstring, score in brute_force_list}
    print(f"Number of solutions: {len(brute_force_results)}")
    print(f"Max objective value: {max(brute_force_results.values())}")
    
    # Test GJE decoding
    gje_counts = test_gje_decoding_cirq(B, v, shots=10000)
    
    # Plot results
    print("\nGenerating comparison plots...")
    plot_cirq_results(brute_force_results, gje_counts)
    
    print("\nTest completed! Results saved to 'cirq_test_results.png'")
    print("\nNote: BP decoding is not yet implemented for Cirq.")


if __name__ == "__main__":
    main() 