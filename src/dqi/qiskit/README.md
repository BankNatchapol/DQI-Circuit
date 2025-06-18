# Qiskit Directory

This directory is reserved for Qiskit-specific implementations and optimizations.

## Current Status

The main Qiskit implementation remains in the parent `dqi/` directory for backward compatibility. This includes:

- `initialization/` - State preparation gates
- `dicke_state_preparation/` - Dicke state gates
- `decoding/` - Decoding algorithms (GJE, BP)
- `utils/` - Utility functions

## Future Plans

This directory will contain:

- Qiskit-specific optimizations
- Hardware backend interfaces
- Pulse-level implementations
- Custom transpiler passes
- Qiskit Runtime programs

## Migration Strategy

New Qiskit-specific code should be added here, while maintaining the existing API in the parent directory for compatibility. 