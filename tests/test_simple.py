"""Simple test script to verify DQI implementations."""

import os
import sys
import numpy as np

# Add the src directory to Python path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root)

# Test imports
print("Testing imports...")
try:
    # Qiskit imports
    from src.dqi.qiskit.initialization.state_preparation.gates import UnaryAmplitudeEncoding as QiskitUAE
    from src.dqi.qiskit.dicke_state_preparation.gates import UnkGate as QiskitUnk
    from src.dqi.qiskit.decoding.gates import GJEGate as QiskitGJE
    print("✓ Qiskit imports successful")
except Exception as e:
    print(f"✗ Qiskit import error: {e}")

try:
    # Cirq imports
    from src.dqi.cirq.initialization.state_preparation.gates import UnaryAmplitudeEncoding as CirqUAE
    from src.dqi.cirq.dicke_state_preparation.gates import UnkGate as CirqUnk
    from src.dqi.cirq.decoding.gates import GJEGate as CirqGJE
    print("✓ Cirq imports successful")
except Exception as e:
    print(f"✗ Cirq import error: {e}")

# Test matrix
H = np.array([
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 1],
    [1, 0, 0, 1]
])

print(f"\nTest matrix H shape: {H.shape}")
print("Matrix H:")
print(H)

# Test Qiskit gates
print("\n" + "="*40)
print("Testing Qiskit Gates")
print("="*40)

try:
    # Test UAE
    weights = np.array([0.5, 0.5, 0.5, 0.5])
    uae_gate = QiskitUAE(4, weights)
    print(f"✓ Created Qiskit UAE gate with {uae_gate.num_bit} qubits")
    
    # Test Unk
    unk_gate = QiskitUnk(4, 2)
    print(f"✓ Created Qiskit Unk gate with n={unk_gate.n}, k={unk_gate.k}")
    
    # Test GJE
    gje_gate = QiskitGJE(H)
    print(f"✓ Created Qiskit GJE gate with {len(gje_gate.operations)} operations")
    
except Exception as e:
    print(f"✗ Qiskit gate error: {e}")

# Test Cirq gates
print("\n" + "="*40)
print("Testing Cirq Gates")
print("="*40)

try:
    # Test UAE
    weights = np.array([0.5, 0.5, 0.5, 0.5])
    uae_gate = CirqUAE(4, weights)
    print(f"✓ Created Cirq UAE gate with {uae_gate.num_bit} qubits")
    
    # Test Unk
    unk_gate = CirqUnk(4, 2)
    print(f"✓ Created Cirq Unk gate with n={unk_gate.n}, k={unk_gate.k}")
    
    # Test GJE
    gje_gate = CirqGJE(H)
    print(f"✓ Created Cirq GJE gate with {len(gje_gate.operations)} operations")
    
except Exception as e:
    print(f"✗ Cirq gate error: {e}")

print("\n" + "="*40)
print("Basic functionality test completed!")
print("="*40) 