# DQI‑Circuit: Decoded Quantum Interferometry Implementation
<div>
    <a href="https://arxiv.org/abs/2504.18334"><img src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg"></a>
    <a href="https://github.com/BankNatchapol/DQI-Circuit"><img src="https://img.shields.io/badge/README-GitHub-blue"></a>
</div>
<br>

This repository provides an implementation of the **Decoded Quantum Interferometry (DQI)** algorithm originally introduced in [arXiv:2408.08292](https://arxiv.org/abs/2408.08292). Detailed implementation choices, performance benchmarks, and extensions are described in our paper ["Quantum Circuit Design for Decoded Quantum Interferometry" (arXiv:2504.18334)](https://arxiv.org/abs/2504.18334).

---

## Features

- **Multiple Decoding Methods**:
  - Belief Propagation Quantum Matching (BPQM)
  - Gauss-Jordan Elimination
  - Unambiguous State Discrimination (USD)
  
- **Circuit Generation**:
  - Automated circuit generation for different decoding methods
  - Support for custom parity check matrices
  - Random matrix generation with controllable density
  
- **Analysis Tools**:
  - Circuit depth and qubit count analysis
  - Gate count statistics
  - Performance benchmarking
  - Visualization utilities

---

## Repository Layout

```
.
├── README.md                   # This overview
├── requirements.txt            # Pinned dependencies
├── src/
│   └── dqi/                   # Main package
│       ├── qiskit/            # Qiskit-specific implementations
│       │   ├── initialization/ # Qiskit state preparation and weight calculation
│       │   ├── dicke_state_preparation/ # Qiskit Dicke state gates
│       │   └── decoding/      # Qiskit decoding gates, algorithms, and BPQM
│       ├── cirq/              # Cirq-specific implementations
│       │   ├── initialization/ # Cirq state preparation and weight calculation
│       │   ├── dicke_state_preparation/ # Cirq Dicke state gates
│       │   └── decoding/      # Cirq decoding gates and algorithms
│       └── utils/             # Shared utilities and helper functions
├── scripts/                   # Command‑line tools and circuit generators
│   ├── generate_dqi_circuit_qiskit.py  # Qiskit circuit generation
│   └── generate_dqi_circuit_cirq.py    # Cirq circuit generation
├── notebooks/                 # Interactive demos & resource estimation
│   ├── main.ipynb            # Main DQI implementation
│   ├── main_bp.ipynb         # Belief propagation implementation
│   └── main_RSB.ipynb        # Reed-Solomon-Bacon implementation
├── figures/                   # Generated figures and plots
├── docs/                      # Documentation
└── tests/                     # Test files
    ├── test_qiskit_implementation.py  # Qiskit tests
    └── test_cirq_implementation.py    # Cirq tests
```

### Multi-Library Support

The codebase now supports multiple quantum computing frameworks:

- **Qiskit Implementation** (`src/dqi/qiskit/`): Original implementation using IBM's Qiskit
- **Cirq Implementation** (`src/dqi/cirq/`): Google Cirq implementation with equivalent functionality
- **Common Code** (`src/dqi/common/`): Shared algorithms and utilities

To use a specific framework:

```python
# For Qiskit
from scripts.generate_dqi_circuit_qiskit import generate_dqi_circuit

# For Cirq
from scripts.generate_dqi_circuit_cirq import generate_dqi_circuit_cirq
```

Both implementations generate compatible circuits with similar interfaces but leverage framework-specific optimizations.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/BankNatchapol/DQI-Circuit.git
cd DQI-Circuit

python -m venv dqi_env
# On Unix/macOS:
source dqi_env/bin/activate
# On Windows cmd.exe:
dqi_env\Scripts\activate.bat
# On Windows PowerShell:
dqi_env\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Clone the BPQM syndrome decoding repository
git clone https://github.com/BankNatchapol/BPQM-Syndrome-Decoding src/dqi/decoding/BPQM

# Note: If you get an error about the directory already existing, you can remove it first:
# rm -rf src/dqi/decoding/BPQM
```

---

## Usage

### Basic Circuit Generation

```python
# For Qiskit implementation
from scripts.generate_dqi_circuit_qiskit import generate_dqi_circuit

# For Cirq implementation  
from scripts.generate_dqi_circuit_cirq import generate_dqi_circuit_cirq

# Generate circuit with random matrix (Qiskit)
circuit, properties = generate_dqi_circuit(
    n=8,                     # number of variable nodes
    m=4,                     # number of check nodes
    density=0.5,            # density of 1s in the matrix
    save_path="circuits/random_circuit.qasm"  # optional: save the circuit
)

# Generate circuit with specific matrix (Cirq)
import numpy as np
H = np.array([
    [1, 1, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 1]
])
circuit, properties = generate_dqi_circuit_cirq(
    H=H,
    save_path="circuits/specific_circuit.qasm"
)
```

### Batch Circuit Generation

To generate multiple DQI circuits with different sizes and decoding methods:

```bash
# Generate circuits using both BP and GJE decoding methods
python scripts/generate_dqi_circuit_qiskit.py
```

The script will generate circuits for various matrix sizes (from 4x4 up to 24x24) using both Belief Propagation (BP) and Gauss-Jordan Elimination (GJE) methods. You can customize the generation by modifying the following parameters in `scripts/generate_dqi_circuit_qiskit.py`:

- `circuit_sizes`: List of tuples defining matrix dimensions (n,m)
- `max_attempts`: Number of retries per size (default: 3)
- `timeout_seconds`: Timeout per attempt (default: 300 seconds)
- `methods`: List of decoding methods to use (["bp", "gje"])
- `max_height`: Maximum BP height to try before giving up (default: 4)

Example of customizing circuit sizes:
```python
# In scripts/generate_dqi_circuit_qiskit.py
circuit_sizes = [(4,4), (6,6), (8,8), (10,10), 
                 (12,12), (14,14), (16,16), (18,18), (20,20),
                 (22,22), (24,24)]  # Modify this list to generate different sizes
```

The script will generate circuits with the following characteristics:

1. Circuit Generation Parameters:
   - Matrix sizes from 4x4 up to 24x24
   - Maximum 3 attempts per size
   - 300 seconds timeout per attempt
   - BP height parameter from 2 to 4
   - Both BP (Belief Propagation) and GJE (Gauss-Jordan Elimination) methods
   - Random matrix density of 0.1

2. Output Structure:
```
circuits/
├── bp/                           # BP decoded circuits
│   └── circuit_YYYYMMDD_HHMMSS_n8m8_q24_d42_bp/  # Directory name format explained below
│       ├── circuit.qpy          # Circuit in QPY format
│       ├── circuit_properties.json  # Detailed gate counts etc.
│       ├── metadata.json        # Circuit parameters and configuration
│       └── parity_check_matrix.npy  # Matrix H used
└── gje/                          # GJE decoded circuits
    └── circuit_YYYYMMDD_HHMMSS_n8m8_q24_d42_gje/
        ├── circuit.qpy
        ├── circuit_properties.json
        ├── metadata.json
        └── parity_check_matrix.npy
```

Directory Name Format:
- `circuit_`: Prefix for all generated circuits
- `YYYYMMDD_HHMMSS`: Timestamp of generation (e.g., 20240418_143022)
- `n8m8`: Matrix dimensions (n=columns, m=rows)
- `q24`: Total number of qubits in the circuit
- `d42`: Circuit depth after transpilation
- `bp` or `gje`: Decoding method used

Example: `circuit_20240418_143022_n8m8_q24_d42_bp` means:
- Generated on April 18, 2024 at 14:30:22
- 8x8 parity check matrix
- Uses 24 qubits
- Has depth 42 after transpilation
- Uses Belief Propagation decoding

The metadata includes:
```json
{
    "creation_date": "YYYYMMDD_HHMMSS",
    "circuit_size": "8x8",
    "num_qubits": 24,
    "circuit_depth": 42,
    "gate_counts": {
        "ecr": 30,
        "rz": 40,
        "sx": 20,
        "x": 10,
        "measure": 16
    },
    "matrix_density": 0.1,
    "decoding_method": "bp",
    "basis_gates": ["ecr", "id", "rz", "sx", "x"],
    "optimization_level": 3
}
```

Important Notes:
- The script uses the same matrix H for both BP and GJE methods
- For BP, it tries increasing heights (2 to 4) until success
- Only successful circuits are saved
- Previous circuits of the same size and method are automatically removed
- All circuits are transpiled to basis gates (ECR, ID, RZ, SX, X) with optimization level 3

### Interactive Notebooks

- `notebooks/main.ipynb`: Main DQI implementation and examples
- `notebooks/main_bp.ipynb`: Belief propagation implementation and analysis
- `notebooks/main_RSB.ipynb`: Reed-Solomon-Bacon code implementation

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

```bibtex
@misc{patamawisut2025quantumcircuitdesigndecoded,
      title={Quantum Circuit Design for Decoded Quantum Interferometry}, 
      author={Natchapol Patamawisut and Naphan Benchasattabuse and Michal Hajdušek and Rodney Van Meter},
      year={2025},
      eprint={2504.18334},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2504.18334}, 
}
```

---

## Progress
### Completed ✅
- Deterministic Dicke state preparation  
- Gauss–Jordan elimination decoder  
- Scalable BPQM implementation for syndrome decoding  
  - Works on most matrices (may not produce correct results for irregular matrices)  
- Integration of BPQM with DQI (working)  

### To Do ✎
- Clean up and refactor the code for clarity and proper structure ⏳  
- Analyze the effect of problem structure on BPQM syndrome decoding  
  - e.g. BPQM tends to fail (gives wrong result) when the factor graph has isolated nodes  
- Study the impact of BPQM parameters on DQI performance  
  - prior, theta, and height  
- Perform resource estimation for FTQC Q-Fly  
- Improve implementation of Dicke state preparation  


