# DQI Circuit Tests

This directory contains test scripts for both Qiskit and Cirq implementations of DQI circuits.

## Test Files

### test_qiskit_implementation.py
Tests the Qiskit implementation of DQI circuits with both:
- **Belief Propagation (BP)** decoding
- **Gauss-Jordan Elimination (GJE)** decoding

### test_cirq_implementation.py
Tests the Cirq implementation of DQI circuits with:
- **Gauss-Jordan Elimination (GJE)** decoding
- Note: BP decoding is not yet implemented for Cirq

## Running the Tests

### Qiskit Test
```bash
python tests/test_qiskit_implementation.py
```

This will:
1. Create a test parity check matrix H
2. Compute brute force objective values
3. Run GJE decoding simulation
4. Run BP decoding simulation (may timeout for certain matrices)
5. Generate comparison plots saved as `qiskit_test_results.png`

### Cirq Test
```bash
python tests/test_cirq_implementation.py
```

This will:
1. Create a test parity check matrix H
2. Compute brute force objective values
3. Run GJE decoding simulation
4. Generate comparison plots saved as `cirq_test_results.png`

## Test Matrix

Both tests use the same 4x8 parity check matrix:
```
H = [[1, 1, 0, 0, 1, 0, 0, 0], 
     [0, 1, 1, 0, 0, 1, 0, 0], 
     [0, 0, 1, 1, 0, 0, 1, 0], 
     [1, 0, 0, 1, 0, 0, 0, 1]]
```

## Output Plots

Each test generates a plot with:
1. **Brute Force Results**: True objective values for all possible solutions
2. **Decoding Results**: Probability distribution from quantum simulation
3. **Combined Comparison**: Overlay of both results for visual comparison

## Expected Results

The quantum simulations should show higher probabilities for solutions with higher objective values, demonstrating that the DQI algorithm successfully finds optimal or near-optimal solutions to the Max-XORSAT problem.

## Troubleshooting

- **BP Timeout**: The BP decoding may timeout for certain matrix structures. This is expected behavior.
- **Import Errors**: Make sure you've installed all dependencies and cloned the BPQM repository as specified in the main README.
- **Memory Issues**: For larger matrices, reduce the number of shots in the simulation. 