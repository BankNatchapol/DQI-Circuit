import os
import sys
import numpy as np
from typing import Optional, Tuple, Dict, Union, Any
import json
from qiskit import qpy
from datetime import datetime
import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Add the src directory to Python path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root)

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

from src.dqi.initialization.state_preparation.gates import UnaryAmplitudeEncoding
from src.dqi.initialization.calculate_w import get_optimal_w
from src.dqi.dicke_state_preparation.gates import UnkGate
from src.dqi.decoding.BPQM.linearcode import LinearCode
from src.dqi.decoding.BPQM.decoders import create_init_qc, decode_single_syndrome
from src.dqi.decoding.BPQM.cloner import VarNodeCloner
from src.dqi.utils.graph import get_max_xorsat_matrix
from src.dqi.decoding.gates import GJEGate


def generate_random_matrix(n: int, m: int, density: float = 0.5) -> np.ndarray:
    return (np.random.random((m, n)) < density).astype(int)


def get_circuit_properties(circuit: QuantumCircuit) -> Dict[str, int]:
    """Get basic properties of the quantum circuit."""
    return {
        "num_qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "gate_counts": circuit.count_ops()
    }


def decode_with_timeout(syndrome_qc, code, theta, height, timeout):
    """Run decode_single_syndrome with timeout."""
    result = None
    
    def _decode():
        nonlocal result
        try:
            result = decode_single_syndrome(
                syndrome_qc=syndrome_qc,
                code=code,
                prior=0.5,
                theta=theta,
                height=height,
                shots=1024,
                debug=True,
                run_simulation=False
            )
        except Exception as e:
            print(f"Error during decoding: {str(e)}")
            result = None
    
    # Start decoding in a separate thread
    thread = threading.Thread(target=_decode)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        print(f"Decoding circuit generation timed out after {timeout}s")
        return None
    
    return result


def generate_dqi_circuit(
    H: Optional[np.ndarray] = None,
    n: int = 8,
    m: int = 4,
    density: float = 0.1,
    theta: float = 0.2 * np.pi,
    save_path: Optional[str] = None,
    method: str = "bp",  # "bp" for Belief Propagation or "gje" for Gauss-Jordan Elimination
    height: int = 2,
    timeout: int = 10  # Add timeout parameter
) -> Tuple[QuantumCircuit, Dict[str, Union[int, Dict[str, int]]]]:
    """
    Generate a DQI circuit with either belief propagation or Gauss-Jordan elimination.
    
    Args:
        H: Optional parity check matrix. If None, a random matrix will be generated.
        n: Number of variable nodes (used if H is None) - This is the number of columns in H
        m: Number of check nodes (used if H is None) - This is the number of rows in H
        density: Base density for random matrix generation.
        theta: Angle parameter for belief propagation (only used if method="bp")
        save_path: Optional path to save the circuit
        method: Decoding method to use ("bp" or "gje")
        height: Height parameter for belief propagation (only used if method="bp")
        timeout: Timeout for BP decoding circuit generation
        
    Returns:
        Tuple of (quantum circuit, circuit properties)
    """
    print(f"\nStarting DQI Circuit Generation with {method.upper()} decoding...")
    if method == "bp":
        print(f"Using BP height: {height}")
    
    # Generate or use provided parity check matrix
    if H is None:
        print(f"Generating random {m}x{n} parity check matrix H with density {density}")
        H = generate_random_matrix(n, m, density)  # n columns, m rows
    else:
        m, n = H.shape
        print(f"Using provided {m}x{n} parity check matrix H")
    
    print(f"Matrix H shape: {H.shape}")
    print("Matrix H contents:\n", H)
    
    # For constraint operations, we need B = H^T
    # So B will be n x m (n rows, m columns)
    B = H.T
    print(f"Matrix B shape (H transposed): {B.shape}")
    print("Matrix B contents:\n", B)
    
    # Print detailed matrix info for small matrices
    if n < 10:
        print("\nDetailed Matrix Analysis:")
        print("H matrix:")
        for i in range(m):
            row = " ".join(str(x) for x in H[i])
            ones = sum(H[i])
            print(f"Row {i}: [{row}] (weight: {ones})")
        
        print("\nB matrix (H transposed):")
        for i in range(n):
            row = " ".join(str(x) for x in B[i])
            ones = sum(B[i])
            print(f"Row {i}: [{row}] (weight: {ones})")
        
        total_ones = np.sum(H)
        density = total_ones / (n * m)
        print(f"\nTotal 1s: {total_ones}")
        print(f"Actual density: {density:.3f}")
        print()
    
    # Get optimal weights for initialization
    print("\nCalculating optimal weights...")
    p, r, ell = 2, 1, 2
    w = get_optimal_w(m, ell, p, r)
    print("Optimal weights:", w)
    
    # Create registers
    print("\nCreating quantum and classical registers...")
    dicke_qregs = QuantumRegister(m, name='y')
    dicke_cregs = ClassicalRegister(m, name='cy')
    syndrome_qregs = QuantumRegister(n, name='syndrome')
    syndrome_cregs = ClassicalRegister(n, name='csolution')
    print(f"Created registers: {m} Dicke qubits (rows of H), {n} syndrome qubits (columns of H)")
    
    # Initialize the circuit
    print("\nInitializing circuit with Unary Amplitude Encoding...")
    init_qregs = QuantumRegister(m, name='k')
    initialize_circuit = QuantumCircuit(init_qregs)
    WUE_Gate = UnaryAmplitudeEncoding(num_bit=m, weights=w)
    initialize_circuit.append(WUE_Gate, range(m))
    
    # Prepare Dicke state circuit
    print("\nPreparing Dicke state circuit...")
    dicke_circuit = QuantumCircuit(dicke_qregs)
    max_errors = int(np.nonzero(w)[0][-1]) if np.any(w) else 0
    dicke_circuit.append(UnkGate(m, max_errors), range(m))
    print(f"Max errors for Dicke state: {max_errors}")
    
    # Setup decoding based on method
    if method == "bp":
        print("\nSetting up belief propagation...")
        cloner = VarNodeCloner(theta)
        code = LinearCode(None, H)
        syndrome_qc = QuantumCircuit(code.hk)
        print(f"Using theta = {theta/np.pi:.3f}π for belief propagation")
        
        # Get decoding circuit with timeout
        print("\nGenerating decoding circuit...")
        decode_result = decode_with_timeout(syndrome_qc, code, theta, height, timeout)
        if decode_result is None:
            raise TimeoutError("BP decoding circuit generation timed out")
        
        _, _, qc_decode = decode_result
        
        # Create ancilla register
        print("\nCreating ancilla register...")
        ancilla_qregs = QuantumRegister(qc_decode.num_qubits - m, name='ancilla')
        print(f"Number of ancilla qubits: {qc_decode.num_qubits - m}")
        
        # Build the complete DQI circuit
        print("\nBuilding complete DQI circuit...")
        dqi_circuit = QuantumCircuit(
            dicke_qregs, syndrome_qregs, syndrome_cregs, 
            dicke_cregs, ancilla_qregs
        )
    else:  # method == "gje"
        print("\nSetting up Gauss-Jordan elimination...")
        # For GJE, we don't need ancilla qubits
        dqi_circuit = QuantumCircuit(
            dicke_qregs, syndrome_qregs, syndrome_cregs, 
            dicke_cregs
        )
    
    # Compose the circuit parts
    print("Composing circuit parts...")
    dqi_circuit.compose(initialize_circuit, inplace=True)
    dqi_circuit.barrier()
    dqi_circuit.compose(dicke_circuit, inplace=True)
    dqi_circuit.barrier()
    
    # Add the constraint matrix operations
    print("\nAdding constraint matrix operations...")
    cx_count = 0
    for i in range(n):
        for j in range(m):
            if B[i][j] == 1:  # Using B[i][j] directly since B is already transposed
                dqi_circuit.cx(j, m+i)
                cx_count += 1
    print(f"Added {cx_count} CNOT gates for constraints")
    
    # Add decoding circuit based on method
    print("\nAdding decoding circuit...")
    if method == "bp":
        dqi_circuit.compose(
            qc_decode,
            qubits=list(range(dicke_qregs.size, qc_decode.num_qubits)) + list(range(dicke_qregs.size)),
            inplace=True
        )
    else:  # method == "gje"
        if n > m:
            gje_circuit = QuantumCircuit(n, name="GJE")
            gje_circuit.append(GJEGate(B.T), list(range(n)))
            dqi_circuit.compose(gje_circuit, qubits=list(range(n)), inplace=True)
        else:
            gje_circuit = QuantumCircuit(m, name="GJE")
            gje_circuit.append(GJEGate(B.T), list(range(n)))
            dqi_circuit.compose(gje_circuit, qubits=list(range(m)), inplace=True)
    
    # Add final Hadamard gates and measurements
    print("\nAdding final Hadamard gates and measurements...")
    for i in range(n):
        dqi_circuit.h(m+i)
    
    dqi_circuit.barrier()
    dqi_circuit.measure(dicke_qregs, dicke_cregs)
    dqi_circuit.measure(syndrome_qregs, syndrome_cregs)
    
    # Transpile circuit to specified basis gates
    print("\nTranspiling circuit to basis gates (ECR, ID, RZ, SX, X)...")
    dqi_circuit = transpile(
        dqi_circuit,
        basis_gates=['ecr', 'id', 'rz', 'sx', 'x'],
        optimization_level=3
    )
    
    # Get circuit properties after transpilation
    print("\nCalculating final circuit properties...")
    properties = get_circuit_properties(dqi_circuit)
    print("Circuit properties:")
    print(f"- Number of qubits: {properties['num_qubits']}")
    print(f"- Circuit depth: {properties['depth']}")
    print("- Gate counts:")
    for gate, count in properties['gate_counts'].items():
        print(f"  * {gate}: {count}")

    # Save circuit if path provided
    if save_path:
        # Create descriptive folder name with transpiled circuit depth
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"circuit_{timestamp}_n{n}m{m}_q{dqi_circuit.num_qubits}_d{dqi_circuit.depth()}_{method}"
        
        # Create folder path
        base_dir = os.path.dirname(save_path)
        new_dir = os.path.join(base_dir, folder_name)  # Only one level of method folder
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            print(f"\nCreated directory: {new_dir}")
        
        # Update save paths with new directory
        qpy_path = os.path.join(new_dir, 'circuit.qpy')
        props_path = os.path.join(new_dir, 'circuit_properties.json')
        matrix_path = os.path.join(new_dir, 'parity_check_matrix.npy')
        
        # Save metadata with transpiled circuit properties
        metadata = {
            "creation_date": timestamp,
            "circuit_size": f"{n}x{m}",
            "num_qubits": dqi_circuit.num_qubits,
            "circuit_depth": dqi_circuit.depth(),
            "gate_counts": properties["gate_counts"],
            "matrix_density": float(np.sum(H) / (n * m)),
            "decoding_method": method,
            "basis_gates": ['ecr', 'id', 'rz', 'sx', 'x'],
            "optimization_level": 3
        }
        metadata_path = os.path.join(new_dir, 'metadata.json')
        
        # Save all files
        with open(qpy_path, 'wb') as f:
            qpy.dump(dqi_circuit, f)
        print(f"Saved QPY file: {qpy_path}")
        
        with open(props_path, 'w') as f:
            json.dump(properties, f, indent=2)
        print(f"Saved properties: {props_path}")
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata: {metadata_path}")
        
        np.save(matrix_path, H)
        print(f"Saved matrix: {matrix_path}")
    
    print("\nCircuit generation complete!")
    return dqi_circuit, properties  # Return transpiled circuit properties


def generate_circuit_with_timeout(n: int, m: int, density: float, save_path: str, method: str, timeout: int = 60, H: Optional[np.ndarray] = None, height: int = 2) -> Optional[Tuple[Any, Dict]]:
    """Generate circuit with timeout. Returns None if generation times out."""
    result = None
    temp_save_path = None  # Store the actual save path
    
    def _generate():
        nonlocal result, temp_save_path
        try:
            # Create a temporary path for intermediate saves
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = os.path.dirname(save_path)
            folder_name = f"circuit_{timestamp}_n{n}m{m}_temp"
            temp_save_path = os.path.join(base_dir, folder_name)
            
            # Generate the circuit with temporary path
            result = generate_dqi_circuit(
                n=n, 
                m=m, 
                density=density, 
                save_path=temp_save_path, 
                method=method, 
                H=H, 
                height=height,
                timeout=timeout  # Pass the timeout parameter
            )
        except Exception as e:
            print(f"Error during circuit generation: {str(e)}")
            result = None
    
    # Start generation in a separate thread
    thread = threading.Thread(target=_generate)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        print(f"Circuit generation timed out after {timeout}s")
        # Clean up temporary directory if it exists
        if temp_save_path and os.path.exists(temp_save_path):
            import shutil
            shutil.rmtree(temp_save_path)
        return None
    
    # If generation was successful, rename the temporary directory to final name
    if result is not None and temp_save_path and os.path.exists(temp_save_path):
        circuit, props = result
        final_folder_name = f"circuit_{timestamp}_n{n}m{m}_q{circuit.num_qubits}_d{circuit.depth()}_{method}"
        final_save_path = os.path.join(base_dir, final_folder_name)
        
        # Remove any existing directory for this size and method
        existing_pattern = f"circuit_*_n{n}m{m}_*_{method}"
        for item in os.listdir(base_dir):
            if item.startswith("circuit_") and f"_n{n}m{m}_" in item and item.endswith(f"_{method}"):
                full_path = os.path.join(base_dir, item)
                if os.path.isdir(full_path):
                    import shutil
                    shutil.rmtree(full_path)
        
        # Rename temporary directory to final name
        os.rename(temp_save_path, final_save_path)
    
    return result


if __name__ == "__main__":
    # Example usage with timeout and retries
    circuit_sizes = [(4,4), (6,6), (8,8), (10,10), 
                     (12,12), (14,14), (16,16), (18,18), (20,20),
                     (22,22), (24,24)]  # Add more sizes as needed
    max_attempts = 3  # Number of retries per size
    timeout_seconds = 300  # Timeout per attempt
    methods = ["bp", "gje"]  # Generate both BP and GJE circuits
    max_height = 4  # Maximum BP height to try before giving up
    
    # Create base directories for each method
    for method in methods:
        os.makedirs(os.path.join("circuits", method), exist_ok=True)
    
    for n, m in circuit_sizes:
        print(f"\nTrying {n}x{m} circuit...")
        
        # Generate random matrix H once for this size
        success = False
        H = None
        for attempt in range(max_attempts):
            if attempt > 0:
                print(f"Matrix generation attempt {attempt + 1}/{max_attempts}...")
            
            # Generate a new random matrix
            H = generate_random_matrix(n, m, density=0.1)
            
            # Verify matrix properties (you can add more checks here)
            if np.sum(H) > 0:  # Basic check that matrix isn't all zeros
                success = True
                break
            
        if not success:
            print(f"Failed to generate valid {n}x{m} matrix after {max_attempts} attempts, skipping to next size.")
            continue
        
        # Use the same matrix H for both methods
        for method in methods:
            print(f"\nGenerating {method.upper()} circuit...")
            success = False
            
            if method == "bp":
                # For BP, try increasing heights until success or max_height
                for height in range(2, max_height + 1):
                    print(f"Trying BP with height {height}...")
                    
                    try:
                        result = generate_circuit_with_timeout(
                            n=n,
                            m=m,
                            density=0.1,
                            save_path=f"circuits/{method}/random_circuit_{n}x{m}.qpy",
                            method=method,
                            timeout=timeout_seconds,
                            H=H,
                            height=height
                        )
                        
                        if result is not None:
                            circuit, props = result
                            print(f"Successfully generated {n}x{m} circuit with {method.upper()} (height={height})!")
                            print("Circuit Properties:", props)
                            success = True
                            break
                    except Exception as e:
                        print(f"Error during circuit generation: {str(e)}")
                    
                    if not success:
                        print(f"BP with height {height} failed, trying next height...")
                        time.sleep(1)  # Brief pause between attempts
                
                if not success:
                    print(f"Failed to generate {n}x{m} circuit with BP after trying heights up to {max_height}")
            else:
                # For GJE, just try the normal way
                for attempt in range(max_attempts):
                    if attempt > 0:
                        print(f"Circuit generation attempt {attempt + 1}/{max_attempts}...")
                    
                    try:
                        result = generate_circuit_with_timeout(
                            n=n,
                            m=m,
                            density=0.1,
                            save_path=f"circuits/{method}/random_circuit_{n}x{m}.qpy",
                            method=method,
                            timeout=timeout_seconds,
                            H=H
                        )
                        
                        if result is not None:
                            circuit, props = result
                            print(f"Successfully generated {n}x{m} circuit with {method.upper()}!")
                            print("Circuit Properties:", props)
                            success = True
                            break
                    except Exception as e:
                        print(f"Error during circuit generation: {str(e)}")
                    
                    if not success and attempt < max_attempts - 1:
                        print("Retrying circuit generation...")
                        time.sleep(1)  # Brief pause between attempts
                
                if not success:
                    print(f"Failed to generate {n}x{m} circuit with {method.upper()} after {max_attempts} attempts") 