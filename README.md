# DQI‑Circuit: Decoded Quantum Interferometry Implementation
<div>
    <a href="https://arxiv.org/abs/2504.18334"><img src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg"></a>
    <a href="https://github.com/BankNatchapol/DQI-Circuit"><img src="https://img.shields.io/badge/README-GitHub-blue"></a>
</div>
<br>

This repository provides an implementation of the **Decoded Quantum Interferometry (DQI)** algorithm originally introduced in [arXiv:2408.08292](https://arxiv.org/abs/2408.08292). Detailed implementation choices, performance benchmarks, and extensions are described in our paper ["Quantum Circuit Design for Decoded Quantum Interferometry" (arXiv:2504.18334)](https://arxiv.org/abs/2504.18334).

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
