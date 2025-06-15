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
│       ├── initialization/    # State preparation and weights calculation
│       ├── dicke_state_preparation/  # Gate for preparing dicke state
│       ├── decoding/         # BPQM, Gauss-Jordan elimination, USD gates
│       │   ├── BPQM/        # Belief Propagation Quantum Matching
│       │   └── gates.py     # Core decoding gates
│       └── utils/           # Timing, plotting, and helper functions
├── scripts/                  # Command‑line tools and circuit generators
│   └── generate_dqi_bp_circuit.py  # BP circuit generation script
├── notebooks/               # Interactive demos & resource estimation
│   ├── main.ipynb          # Main DQI implementation
│   ├── main_bp.ipynb       # Belief propagation implementation
│   └── main_RSB.ipynb      # Reed-Solomon-Bacon implementation
├── figures/                 # Generated figures and plots
├── docs/                    # Documentation
└── tests/                   # Test files
```

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
from scripts.generate_dqi_bp_circuit import generate_dqi_bp_circuit

# Generate circuit with random matrix
circuit, properties = generate_dqi_bp_circuit(
    n=8,                     # number of variable nodes
    m=4,                     # number of check nodes
    density=0.5,            # density of 1s in the matrix
    save_path="circuits/random_circuit.qasm"  # optional: save the circuit
)

# Generate circuit with specific matrix
import numpy as np
H = np.array([
    [1, 1, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 1]
])
circuit, properties = generate_dqi_bp_circuit(
    H=H,
    save_path="circuits/specific_circuit.qasm"
)
```

### Batch Circuit Generation

To generate multiple DQI circuits with different sizes and decoding methods:

```bash
# Generate circuits using both BP and GJE decoding methods
python scripts/generate_dqi_circuit.py
```

The script will generate circuits for various matrix sizes (from 4x4 up to 24x24) using both Belief Propagation (BP) and Gauss-Jordan Elimination (GJE) methods. You can customize the generation by modifying the following parameters in `scripts/generate_dqi_circuit.py`:

- `circuit_sizes`: List of tuples defining matrix dimensions (n,m)
- `max_attempts`: Number of retries per size (default: 3)
- `timeout_seconds`: Timeout per attempt (default: 300 seconds)
- `methods`: List of decoding methods to use (["bp", "gje"])
- `max_height`: Maximum BP height to try before giving up (default: 4)

Example of customizing circuit sizes:
```python
# In scripts/generate_dqi_circuit.py
circuit_sizes = [(4,4), (6,6), (8,8), (10,10), 
                 (12,12), (14,14), (16,16), (18,18), (20,20),
                 (22,22), (24,24)]  # Modify this list to generate different sizes
```

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


