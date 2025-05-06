# DQI‑Circuit: Decoded Quantum Interferometry Implementation
<div>
    <a href="https://arxiv.org/abs/2504.18334"><img src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg"></a>
    <a href="https://github.com/BankNatchapol/DQI-Circuit"><img src="https://img.shields.io/badge/README-GitHub-blue"></a>
</div>
<br>

This repository provides an implementation of the **Decoded Quantum Interferometry (DQI)** algorithm originally introduced in [arXiv:2408.08292](https://arxiv.org/abs/2408.08292). Detailed implementation choices, performance benchmarks, and extensions are described in our paper ["Quantum Circuit Design for Decoded Quantum Interferometry" (arXiv:2504.18334)](https://arxiv.org/abs/2504.18334).

---

## Key Features

* **Core DQI Circuit**: End‑to‑end Qiskit implementation of the DQI decoding circuit, matching the gate sequence from the original algorithm.
* **State Preparation & Decoding Utilities**: Modular routines for amplitude encoding, Gauss–Jordan elimination gates, and Unambiguous State Discrimination (USD)‑based reduction.
* **Eigenvector Computations**: Dense and sparse methods for principal eigenvector extraction, with automatic padding to power‑of‑two dimensions.
* **Max‑XORSAT Solvers**: Brute‑force and DQI‑based solvers for small to medium QUBO instances, with plotting utilities for comparison.
* **Benchmarking & Examples**: Scripts and notebooks to reproduce timing plots, resource estimates, and optimization benchmarks.
* **Flexible Backends**: Runs on Qiskit AerSimulator and easily configured for IBM Quantum hardware.

---

## Repository Layout

```text
.
├── README.md                   # This overview
├── requirements.txt            # Pinned dependencies
├── src/
│   └── dqi/                    # Main package
│       ├── initialization/     # State preparation and weights calculation
│       ├── dicke_state_preparation/     # Gate for preparing dicke state
│       ├── decoding/           # Gauss-Jordan elimination, USD, and solver gates
│       └── utils/              # Timing, plotting, and helper functions
├── scripts/                    # Command‑line benchmarks & runners
├── notebooks/                  # Interactive demos & resource estimation
│   └── main.ipynb
├── figures/                    
├── docs/                    
└── tests/                      # Random testing files
```

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/yourusername/dqi-circuit.git
cd dqi-circuit
pip install -e .
```

### 2. Dependencies

* Python 3.8+
* Qiskit (`qiskit`, `qiskit-aer`)
* NumPy, SciPy, Plotly, NetworkX, Matplotlib

All dependencies are pinned in `requirements.txt`.

### 3. Basic Example

```python
from dqi.decoding.gates import GJEGate
from dqi.initialization.construct_A import construct_A_matrix
from dqi.utils.viz import plot_results_union_plotly

# Build A and w
A_sparse = construct_A_matrix(m=100_000, ell=10, p=2, r=1)
# Create Gauss–Jordan gate
gje = GJEGate(A_sparse.toarray())

# Compare brute force vs DQI for a small graph
# (See examples/decode_example.py for a full workflow)
```

### 4. Run Benchmarks

```bash
python scripts/bench_calculate_w.py
python scripts/bench_construct_A.py
```

### 5. Tests and Coverage

```bash
pytest --cov=src/dqi
```

---

## Contributing

Please see [docs/usage\_guide.md](docs/usage_guide.md) for development guidelines, coding style, and testing conventions. Pull requests should include new tests and update `CHANGELOG.md` as needed.
