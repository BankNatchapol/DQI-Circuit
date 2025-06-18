"""Generate DQI circuits using Cirq quantum computing framework."""

import os
import sys
import numpy as np
from typing import Optional, Tuple, Dict, Union, Any, List
import json
from datetime import datetime
import threading
import time
import cirq

# Add the src directory to Python path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root)

from src.dqi.cirq.initialization.state_preparation.gates import UnaryAmplitudeEncoding
from src.dqi.cirq.initialization.calculate_w import get_optimal_w
from src.dqi.cirq.dicke_state_preparation.gates import UnkGate
from src.dqi.cirq.decoding.gates import GJEGate


def generate_random_matrix(n: int, m: int, density: float = 0.5) -> np.ndarray:
    """Generate a random binary matrix."""
    return (np.random.random((m, n)) < density).astype(int)


def get_circuit_properties(circuit: cirq.Circuit) -> Dict[str, Any]:
    """Get basic properties of the Cirq circuit."""
    # Count operations
    gate_counts = {}
    for moment in circuit:
        for op in moment:
            gate_type = type(op.gate).__name__
            if gate_type in gate_counts:
                gate_counts[gate_type] += 1
            else:
                gate_counts[gate_type] = 1
    
    # Get circuit statistics
    stats = cirq.optimizers.compute_circuit_statistics(circuit)
    
    return {
        "num_qubits": len(circuit.all_qubits()),
        "depth": len(circuit),
        "gate_counts": gate_counts,
        "two_qubit_depth": stats.two_qubit_depth
    }


def generate_dqi_circuit_cirq(
    H: Optional[np.ndarray] = None,
    n: int = 8,
    m: int = 4,
    density: float = 0.1,
    save_path: Optional[str] = None,
    method: str = "gje",  # Only GJE is implemented for now
    optimize: bool = True
) -> Tuple[cirq.Circuit, Dict[str, Any]]:
    """
    Generate a DQI circuit using Cirq with Gauss-Jordan elimination.
    
    Args:
        H: Optional parity check matrix. If None, a random matrix will be generated.
        n: Number of variable nodes (columns in H)
        m: Number of check nodes (rows in H)
        density: Base density for random matrix generation.
        save_path: Optional path to save the circuit
        method: Decoding method to use (only "gje" supported)
        optimize: Whether to optimize the circuit
        
    Returns:
        Tuple of (quantum circuit, circuit properties)
    """
    print(f"\nStarting DQI Circuit Generation with Cirq ({method.upper()} decoding)...")
    
    # Generate or use provided parity check matrix
    if H is None:
        print(f"Generating random {m}x{n} parity check matrix H with density {density}")
        H = generate_random_matrix(n, m, density)
    else:
        m, n = H.shape
        print(f"Using provided {m}x{n} parity check matrix H")
    
    print(f"Matrix H shape: {H.shape}")
    print("Matrix H contents:\n", H)
    
    # For constraint operations, we need B = H^T
    B = H.T
    print(f"Matrix B shape (H transposed): {B.shape}")
    
    # Get optimal weights for initialization
    print("\nCalculating optimal weights...")
    p, r, ell = 2, 1, 2
    w = get_optimal_w(m, ell, p, r)
    print("Optimal weights:", w)
    
    # Create qubits
    print("\nCreating qubits...")
    dicke_qubits = [cirq.NamedQubit(f'y_{i}') for i in range(m)]
    syndrome_qubits = [cirq.NamedQubit(f'syndrome_{i}') for i in range(n)]
    all_qubits = dicke_qubits + syndrome_qubits
    print(f"Created {m} Dicke qubits and {n} syndrome qubits")
    
    # Initialize the circuit
    circuit = cirq.Circuit()
    
    # Add Unary Amplitude Encoding
    print("\nAdding Unary Amplitude Encoding...")
    uae_gate = UnaryAmplitudeEncoding(num_bit=m, weights=w)
    circuit.append(uae_gate(*dicke_qubits))
    
    # Add barrier (using moment separation)
    circuit.append(cirq.Moment())
    
    # Add Dicke state preparation
    print("\nAdding Dicke state preparation...")
    max_errors = int(np.nonzero(w)[0][-1]) if np.any(w) else 0
    unk_gate = UnkGate(m, max_errors)
    circuit.append(unk_gate(*dicke_qubits))
    print(f"Max errors for Dicke state: {max_errors}")
    
    # Add barrier
    circuit.append(cirq.Moment())
    
    # Add constraint matrix operations
    print("\nAdding constraint matrix operations...")
    cx_count = 0
    for i in range(n):
        for j in range(m):
            if B[i][j] == 1:
                circuit.append(cirq.CNOT(dicke_qubits[j], syndrome_qubits[i]))
                cx_count += 1
    print(f"Added {cx_count} CNOT gates for constraints")
    
    # Add decoding circuit
    print("\nAdding GJE decoding circuit...")
    if method == "gje":
        if n > m:
            gje_gate = GJEGate(B.T)
            circuit.append(gje_gate(*syndrome_qubits))
        else:
            # Need to handle case where we have fewer syndrome qubits than dicke qubits
            gje_gate = GJEGate(B.T)
            # Apply to the first n qubits
            circuit.append(gje_gate(*dicke_qubits[:n]))
    else:
        raise NotImplementedError(f"Method {method} not implemented for Cirq")
    
    # Add final Hadamard gates
    print("\nAdding final Hadamard gates...")
    for qubit in syndrome_qubits:
        circuit.append(cirq.H(qubit))
    
    # Add barrier
    circuit.append(cirq.Moment())
    
    # Add measurements
    print("\nAdding measurements...")
    circuit.append([cirq.measure(q, key=f'c{q.name}') for q in dicke_qubits])
    circuit.append([cirq.measure(q, key=f'c{q.name}') for q in syndrome_qubits])
    
    # Optimize circuit if requested
    if optimize:
        print("\nOptimizing circuit...")
        # Apply various Cirq optimizers
        circuit = cirq.merge_single_qubit_gates_to_phased_x_and_z(circuit)
        circuit = cirq.eject_phased_paulis(circuit)
        circuit = cirq.eject_z(circuit)
        circuit = cirq.drop_negligible_operations(circuit)
        circuit = cirq.drop_empty_moments(circuit)
    
    # Get circuit properties
    print("\nCalculating final circuit properties...")
    properties = get_circuit_properties(circuit)
    print("Circuit properties:")
    print(f"- Number of qubits: {properties['num_qubits']}")
    print(f"- Circuit depth: {properties['depth']}")
    print(f"- Two-qubit depth: {properties['two_qubit_depth']}")
    print("- Gate counts:")
    for gate, count in properties['gate_counts'].items():
        print(f"  * {gate}: {count}")
    
    # Save circuit if path provided
    if save_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"circuit_{timestamp}_n{n}m{m}_q{properties['num_qubits']}_d{properties['depth']}_{method}_cirq"
        
        base_dir = os.path.dirname(save_path)
        new_dir = os.path.join(base_dir, folder_name)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            print(f"\nCreated directory: {new_dir}")
        
        # Save circuit as JSON
        circuit_path = os.path.join(new_dir, 'circuit.json')
        with open(circuit_path, 'w') as f:
            cirq.to_json(circuit, f)
        print(f"Saved circuit: {circuit_path}")
        
        # Save circuit as QASM
        qasm_path = os.path.join(new_dir, 'circuit.qasm')
        with open(qasm_path, 'w') as f:
            f.write(cirq.qasm(circuit))
        print(f"Saved QASM: {qasm_path}")
        
        # Save properties
        props_path = os.path.join(new_dir, 'circuit_properties.json')
        with open(props_path, 'w') as f:
            json.dump(properties, f, indent=2)
        print(f"Saved properties: {props_path}")
        
        # Save metadata
        metadata = {
            "creation_date": timestamp,
            "circuit_size": f"{n}x{m}",
            "num_qubits": properties['num_qubits'],
            "circuit_depth": properties['depth'],
            "two_qubit_depth": properties['two_qubit_depth'],
            "gate_counts": properties["gate_counts"],
            "matrix_density": float(np.sum(H) / (n * m)),
            "decoding_method": method,
            "framework": "cirq",
            "optimized": optimize
        }
        metadata_path = os.path.join(new_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata: {metadata_path}")
        
        # Save matrix
        matrix_path = os.path.join(new_dir, 'parity_check_matrix.npy')
        np.save(matrix_path, H)
        print(f"Saved matrix: {matrix_path}")
    
    print("\nCircuit generation complete!")
    return circuit, properties


def main():
    """Main function to demonstrate circuit generation."""
    # Example circuit sizes
    circuit_sizes = [(4,4), (6,6), (8,8), (10,10)]
    
    # Create output directory
    os.makedirs("circuits/cirq", exist_ok=True)
    
    for n, m in circuit_sizes:
        print(f"\n{'='*60}")
        print(f"Generating {n}x{m} circuit...")
        print(f"{'='*60}")
        
        try:
            circuit, props = generate_dqi_circuit_cirq(
                n=n,
                m=m,
                density=0.1,
                save_path=f"circuits/cirq/random_circuit_{n}x{m}.json",
                method="gje",
                optimize=True
            )
            
            print(f"\nSuccessfully generated {n}x{m} circuit!")
            
            # Optionally print the circuit for small sizes
            if n <= 6:
                print("\nCircuit diagram:")
                print(circuit)
                
        except Exception as e:
            print(f"Error generating {n}x{m} circuit: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main() 