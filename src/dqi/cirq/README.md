# Cirq Implementation of DQI Circuits

This directory contains the Cirq implementation of Decoded Quantum Interferometry (DQI) circuits.

## Overview

The Cirq implementation provides equivalent functionality to the Qiskit version but uses Google's Cirq quantum computing framework. This allows for:

- Integration with Google's quantum hardware and simulators
- Different optimization strategies
- Alternative circuit representations
- Compatibility with Cirq's ecosystem

## Structure

```
cirq/
├── __init__.py
├── initialization/
│   └── state_preparation/
│       └── gates.py          # UnaryAmplitudeEncoding gate
├── dicke_state_preparation/
│   └── gates.py             # Dicke state preparation gates
├── decoding/
│   └── gates.py             # GJE decoding gate
└── README.md                # This file
```

## Key Differences from Qiskit Implementation

1. **Gate Implementation**: 
   - Uses `cirq.Gate` base class instead of `qiskit.circuit.Gate`
   - Implements `_decompose_()` instead of `_define()`
   - Uses `_num_qubits_()` and `_circuit_diagram_info_()` for metadata

2. **Circuit Construction**:
   - Uses `cirq.Circuit` instead of `QuantumCircuit`
   - Qubits are `cirq.NamedQubit` objects
   - No separate classical registers (measurements store results directly)

3. **Optimization**:
   - Uses Cirq's built-in optimizers like `merge_single_qubit_gates_to_phased_x_and_z`
   - Different optimization strategies available

4. **Serialization**:
   - Saves circuits as JSON using `cirq.to_json()`
   - Also exports to QASM for compatibility

## Usage Example

```python
from scripts.generate_dqi_circuit_cirq import generate_dqi_circuit_cirq

# Generate a 8x4 circuit
circuit, properties = generate_dqi_circuit_cirq(
    n=8,
    m=4,
    density=0.1,
    save_path="circuits/cirq/my_circuit.json",
    method="gje",
    optimize=True
)

# Print circuit properties
print(f"Number of qubits: {properties['num_qubits']}")
print(f"Circuit depth: {properties['depth']}")
print(f"Two-qubit depth: {properties['two_qubit_depth']}")
```

## Features

- **Unary Amplitude Encoding**: Prepares weighted superposition states
- **Dicke State Preparation**: Efficient preparation of Dicke states
- **GJE Decoding**: Gauss-Jordan elimination for syndrome decoding
- **Circuit Optimization**: Built-in optimization passes
- **Multiple Export Formats**: JSON, QASM, and circuit diagrams

## Requirements

```bash
pip install cirq
pip install numpy
```

## Limitations

- Currently only supports GJE decoding (BP decoding not yet implemented)
- Some advanced features from Qiskit version may not be available

## Future Work

- Implement Belief Propagation decoding for Cirq
- Add support for noise models
- Integrate with Cirq's quantum virtual machine
- Add support for Google's quantum hardware backends 