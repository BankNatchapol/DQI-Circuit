"""Test Qiskit implementation of BPQM decoding - comprehensive tests based on main.ipynb."""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qiskit import QuantumCircuit  # This is the actual Qiskit package
from qiskit_impl.linearcode import LinearCode
from qiskit_impl.cloner import VarNodeCloner, ExtendedVarNodeCloner
from qiskit_impl.decoders import (
    decode_bpqm, 
    decode_single_codeword, 
    decode_single_syndrome, 
    create_init_qc
)


def test_84_code_bpqm():
    """Test BPQM decoding on [8,4] code from Section 6 of the paper."""
    print("\n" + "="*60)
    print("Test 1: [8,4] Code BPQM Decoding (Qiskit)")
    print("="*60)
    
    # Define the [8,4] code
    G = np.array([
        [1, 0, 0, 0, 1, 0, 0, 1],
        [0, 1, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 1, 1],
    ])
    H = np.array([
        [1, 1, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 0, 1],
    ])
    
    code = LinearCode(G, H)
    theta = 0.2 * np.pi
    cloner = VarNodeCloner(theta)
    
    # Test single-bit decode (bit index 4)
    p_bit = decode_bpqm(
        code,
        theta,
        cloner=cloner,
        height=2,
        mode='bit',
        bit=4,
        only_zero_codeword=True,
        debug=False
    )
    print(f"Success probability for bit 4: {p_bit:.6f}")
    print(f"Expected: 0.8602192421422509")
    assert abs(p_bit - 0.8602192421422509) < 0.001, f"Expected ~0.8602, got {p_bit}"
    
    # Test full-codeword decode
    p_codeword = decode_bpqm(
        code,
        theta,
        cloner=cloner,
        height=2,
        mode='codeword',
        order=[0, 1, 2, 3],
        only_zero_codeword=True,
        debug=False
    )
    print(f"Success probability for full codeword: {p_codeword:.6f}")
    print(f"Expected: 0.6893367460101051")
    assert abs(p_codeword - 0.6893367460101051) < 0.001, f"Expected ~0.6893, got {p_codeword}"
    print("✓ Success probabilities match expected values!")


def test_codeword_decoding():
    """Test decoding a specific corrupted codeword - from main.ipynb cell 3."""
    print("\n" + "="*60)
    print("Test 2: Corrupted Codeword Decoding (Qiskit)")
    print("="*60)
    
    # Define code
    G = np.array([
        [1, 0, 0, 0, 1, 0, 0, 1],
        [0, 1, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 1, 1],
    ])
    H = np.array([
        [1, 1, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 0, 1],
    ])
    
    code = LinearCode(None, H)
    theta = 0.2 * np.pi
    cloner = VarNodeCloner(theta)
    
    # Create corrupted codeword
    message = np.array([0, 0, 1, 1])
    codeword = message @ G % 2
    corrupted_codeword = codeword.copy()
    error_position = 3
    corrupted_codeword[error_position] = 1 - corrupted_codeword[error_position]
    
    print(f"Message:             {message}")
    print(f"Original codeword:   {codeword}")
    print(f"Corrupted codeword:  {corrupted_codeword}")
    print(f"Error at position:   {error_position}")
    
    syndrome = (corrupted_codeword @ H.T) % 2
    print(f"Syndrome:            {syndrome}")
    
    # Create initialization circuit
    qc_init = create_init_qc(
        code=code,
        codeword=corrupted_codeword,
        theta=theta,
        prior=None
    )
    
    # Decode with BPQM
    decoded_bits, decoded_qubits, qc_decode = decode_single_codeword(
        qc_init=qc_init,
        code=code,
        cloner=cloner,
        height=2,
        shots=1024,
        debug=False,
        run_simulation=True
    )
    
    print(f"Decoded bits:        {decoded_bits}")
    print(f"Syndrome check:      {(decoded_bits @ H.T) % 2}")
    
    # Check if decoding was successful
    if np.array_equal(decoded_bits, codeword):
        print("✓ Decoding successful - recovered original codeword!")
    else:
        print("✗ Decoding failed")


def test_syndrome_decoding():
    """Test syndrome-based decoding - from main.ipynb cells 5-8."""
    print("\n" + "="*60)
    print("Test 3: Syndrome Decoding [5,1] Code (Qiskit)")
    print("="*60)
    
    # [5,1] code from main.ipynb
    H = np.array([
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 1, 0, 1, 1],
    ])
    
    theta = 0.2 * np.pi
    code = LinearCode(None, H)
    
    # Test case from main.ipynb
    corrupted_codeword = np.array([1, 1, 0, 1, 1])
    syndrome = (corrupted_codeword @ H.T) % 2
    
    print(f"Corrupted codeword:  {corrupted_codeword}")
    print(f"Syndrome:            {syndrome}")
    
    # Create syndrome circuit
    syndrome_qc = QuantumCircuit(len(syndrome))
    for i, s in enumerate(syndrome):
        if s == 1:
            syndrome_qc.x(i)
    
    # Decode with prior=0.5
    decoded_bits, decoded_qubits, qc_decode = decode_single_syndrome(
        syndrome_qc=syndrome_qc,
        code=code,
        prior=0.5,
        theta=theta,
        height=2,
        shots=1024,
        debug=False,
        run_simulation=True
    )
    
    print(f"Decoded error pattern: {decoded_bits}")
    if decoded_bits is not None:
        decoded_syndrome = (decoded_bits @ H.T) % 2
        print(f"Decoded syndrome:      {decoded_syndrome}")
        
        # Check if syndrome matches
        if np.array_equal(decoded_syndrome, syndrome):
            print("✓ Error pattern produces correct syndrome!")
        else:
            print("✗ Error pattern produces different syndrome")


def test_factor_graph():
    """Test factor graph construction."""
    print("\n" + "="*60)
    print("Test 4: Factor Graph Construction (Qiskit)")
    print("="*60)
    
    # [8,4] code
    G = np.array([
        [1, 0, 0, 0, 1, 0, 0, 1],
        [0, 1, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 1, 1],
    ])
    H = np.array([
        [1, 1, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 0, 1],
    ])
    
    code = LinearCode(G, H)
    fg = code.get_factor_graph()
    
    print(f"Factor graph nodes: {fg.number_of_nodes()}")
    print(f"Factor graph edges: {fg.number_of_edges()}")
    
    variable_nodes = [n for n in fg.nodes() if fg.nodes[n]['type'] == 'variable']
    check_nodes = [n for n in fg.nodes() if fg.nodes[n]['type'] == 'check']
    output_nodes = [n for n in fg.nodes() if fg.nodes[n]['type'] == 'output']
    
    print(f"Variable nodes ({len(variable_nodes)}): {variable_nodes}")
    print(f"Check nodes ({len(check_nodes)}): {check_nodes}")
    print(f"Output nodes ({len(output_nodes)}): {output_nodes}")
    
    # Test computation graph
    cg, occ, root = code.get_computation_graph("x0", height=2)
    print(f"\nComputation graph for x0 (height=2):")
    print(f"  Nodes: {cg.number_of_nodes()}")
    print(f"  Variable occurrences: {occ}")


def test_different_heights():
    """Test BPQM with different tree heights using syndrome decoding."""
    print("\n" + "="*60)
    print("Test 5: Different Tree Heights with Syndrome Decoding (Qiskit)")
    print("="*60)
    
    # Simple repetition code
    H = np.array([
        [1, 1, 0],
        [0, 1, 1]
    ])
    
    code = LinearCode(None, H)
    theta = 0.15 * np.pi
    
    # Create an error pattern and its syndrome
    error_pattern = np.array([1, 0, 0])  # Error on first bit
    syndrome = (error_pattern @ H.T) % 2
    
    print(f"True error pattern: {error_pattern}")
    print(f"Syndrome: {syndrome}")
    
    syndrome_qc = QuantumCircuit(len(syndrome))
    for i, s in enumerate(syndrome):
        if s == 1:
            syndrome_qc.x(i)
    
    for height in [1, 2, 3]:
        print(f"\nHeight {height}:")
        
        decoded_bits, _, qc = decode_single_syndrome(
            syndrome_qc=syndrome_qc,
            code=code,
            prior=0.8,  # Strong prior favoring 0
            theta=theta,
            height=height,
            shots=256,
            run_simulation=True
        )
        
        if decoded_bits is not None:
            print(f"  Decoded error pattern: {decoded_bits}")
            decoded_syndrome = (decoded_bits @ H.T) % 2
            print(f"  Decoded syndrome: {decoded_syndrome}")
            
            if np.array_equal(decoded_syndrome, syndrome):
                print(f"  ✓ Error pattern produces correct syndrome")
            else:
                print(f"  ✗ Error pattern produces different syndrome")


def run_all_tests():
    """Run all Qiskit BPQM tests."""
    print("Running Comprehensive Qiskit BPQM Tests")
    print("=======================================")
    
    tests = [
        ("84 Code BPQM", test_84_code_bpqm),
        ("Codeword Decoding", test_codeword_decoding),
        ("Syndrome Decoding", test_syndrome_decoding),
        ("Factor Graph", test_factor_graph),
        ("Different Heights", test_different_heights)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ {test_name} test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"Test Summary: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    if success:
        print("\nAll Qiskit BPQM tests passed! ✓")
    else:
        print("\nSome tests failed! ✗")
        sys.exit(1) 