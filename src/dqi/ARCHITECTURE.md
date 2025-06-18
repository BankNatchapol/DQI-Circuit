# DQI Multi-Library Architecture

This document explains the folder structure and design decisions for supporting multiple quantum computing frameworks in the DQI project.

## Overview

The DQI codebase is designed to support multiple quantum computing frameworks while maintaining code reusability and consistency. Currently supported frameworks:

1. **Qiskit** (IBM) - Original implementation
2. **Cirq** (Google) - Alternative implementation
3. **Future**: PennyLane, Braket, Q#, etc.

## Folder Structure

```
src/dqi/
├── __init__.py              # Main package initialization
├── qiskit/                  # Qiskit-specific implementations
│   ├── __init__.py
│   ├── gates/              # Qiskit custom gates
│   └── utils/              # Qiskit-specific utilities
├── cirq/                    # Cirq-specific implementations
│   ├── __init__.py
│   ├── initialization/
│   ├── dicke_state_preparation/
│   └── decoding/
├── common/                  # Shared code across frameworks
│   ├── algorithms.py       # Classical algorithms
│   ├── matrix_utils.py     # Matrix operations
│   └── constants.py        # Shared constants
├── initialization/          # Original Qiskit implementation (legacy)
├── dicke_state_preparation/ # Original Qiskit implementation (legacy)
├── decoding/               # Original Qiskit implementation (legacy)
└── utils/                  # Original utilities (legacy)
```

## Design Principles

### 1. Framework-Specific Implementation
Each quantum framework has its own subdirectory containing:
- Custom gate implementations
- Framework-specific optimizations
- Serialization/deserialization code
- Hardware backend interfaces

### 2. Shared Algorithms
Classical algorithms and utilities are stored in `common/`:
- Matrix operations (e.g., Gauss-Jordan elimination)
- Weight calculations
- Graph algorithms
- Mathematical utilities

### 3. Consistent Interfaces
All framework implementations should provide similar interfaces:
```python
# Qiskit
circuit = generate_dqi_circuit(H, n=8, m=4, method="gje")

# Cirq
circuit = generate_dqi_circuit_cirq(H, n=8, m=4, method="gje")

# Future: PennyLane
circuit = generate_dqi_circuit_pennylane(H, n=8, m=4, method="gje")
```

### 4. Legacy Support
The original Qiskit implementation remains in the root `dqi/` folder for backward compatibility. New Qiskit-specific code should go in `dqi/qiskit/`.

## Adding a New Framework

To add support for a new quantum framework:

1. Create a new directory: `src/dqi/<framework_name>/`
2. Implement the required gates:
   - `UnaryAmplitudeEncoding`
   - `UnkGate` (Dicke state preparation)
   - `GJEGate` (Gauss-Jordan elimination)
3. Create a circuit generation script: `scripts/generate_dqi_circuit_<framework>.py`
4. Add framework to `requirements.txt`
5. Update documentation

## Benefits

1. **Modularity**: Each framework is self-contained
2. **Reusability**: Common algorithms are shared
3. **Flexibility**: Easy to add new frameworks
4. **Comparison**: Can benchmark across frameworks
5. **Hardware Access**: Use each framework's native hardware support

## Example Usage

```python
# Import framework-specific implementations
from src.dqi.qiskit.gates import UnaryAmplitudeEncoding as QiskitUAE
from src.dqi.cirq.initialization.state_preparation.gates import UnaryAmplitudeEncoding as CirqUAE

# Use common algorithms
from src.dqi.common.algorithms import gauss_jordan_operations_general

# Generate circuits in different frameworks
qiskit_circuit = generate_dqi_circuit(H, save_path="circuits/qiskit/")
cirq_circuit = generate_dqi_circuit_cirq(H, save_path="circuits/cirq/")
```

## Future Enhancements

1. **Unified API**: Create a framework-agnostic API
2. **Automatic Translation**: Convert circuits between frameworks
3. **Performance Benchmarking**: Compare implementations
4. **Hardware Abstraction**: Unified interface for different quantum hardware
5. **Plugin System**: Dynamic loading of framework implementations 