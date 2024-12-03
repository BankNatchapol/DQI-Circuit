# DQI Circuit: Implementation of Decoded Quantum Interferometry Algorithm

## Introduction

This repository contains an implementation of the **Decoded Quantum Interferometry (DQI)** algorithm as described in the research paper ["Decoded Quantum Interferometry" (arXiv:2408.08292)](https://arxiv.org/pdf/2408.08292). 

The DQI Circuit algorithm offers an innovative approach to optimization and quantum computing, leveraging quantum interferometry principles. This implementation aims to reproduce the results and provide a framework for further exploration and experimentation.

## Features

- Implementation of the DQI Circuit as outlined in the paper.
- Includes examples for running the algorithm on test cases.
- Modular and extensible codebase for customizing the circuit design and parameters.
- Supports both simulation and real quantum device execution.

## Repository Structure

```
.
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── src/
│   ├── dqi_circuit.py      # Core implementation of the DQI Circuit
│   ├── utils.py            # Helper functions for quantum operations
│   └── examples/
│       ├── example1.py     # Example usage script
│       ├── example2.py     # Advanced parameter settings
│       └── ...
├── tests/
│   ├── test_dqi.py         # Unit tests for the DQI Circuit implementation
│   ├── test_utils.py       # Unit tests for utility functions
├── data/
│   └── inputs/             # Sample input data for experiments
│   └── outputs/            # Generated results and metrics
└── docs/
    ├── paper_summary.md    # Overview and key takeaways from the paper
    ├── usage_guide.md      # Detailed usage instructions
    └── api_reference.md    # API documentation for the core modules
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Qiskit or other quantum libraries (specified in `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/dqi-circuit.git
   cd dqi-circuit
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Run the example script:
   ```bash
   python src/examples/example1.py
   ```

2. Customize the parameters in `dqi_circuit.py` for your experiments.