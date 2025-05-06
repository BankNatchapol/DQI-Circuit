import numpy as np
from itertools import combinations
import warnings
import random
import pprint
import json

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.providers.fake_provider import GenericBackendV2

from initialization.state_preparation.gates import WeightedUnaryEncoding
from initialization.calculate_w import get_optimal_w
from initialization.construct_A import construct_A_matrix
from dicke_state_preparation.gates import UnkStatePreparation
from decoding.gates import GJEGate
from utils import find_graph_with_target_max_solutions, get_max_xorsat_matrix

warnings.filterwarnings("ignore")
pp = pprint.PrettyPrinter(depth=4)

FAKE_BACKEND = GenericBackendV2(30, basis_gates=['z', 'cx', 'ry', 'swap', 'rz', 'rx'])

def get_transpiled_gate_info(circuit: QuantumCircuit, label: str) -> dict:
    transpiled = transpile(circuit, backend=FAKE_BACKEND)
    ops = transpiled.count_ops()
    total_gates = sum(ops.values())
    depth = transpiled.depth()
    one_qubit_count = sum(1 for instr, qargs, _ in transpiled.data if len(qargs) == 1)
    two_qubit_count = sum(1 for instr, qargs, _ in transpiled.data if len(qargs) == 2)
    num_qubits = circuit.num_qubits

    info = {
        "label": label,
        "num_qubits": num_qubits,
        "gate_counts": ops,
        "total_gates": total_gates,
        "depth": depth,
        "one_qubit_gates": one_qubit_count,
        "two_qubit_gates": two_qubit_count
    }
    return info

def syndrome_decode_lookuptable_gate(syndrome_map: dict) -> QuantumCircuit:
    first_key = next(iter(syndrome_map.keys()))
    n_syndrome = len(first_key)
    first_val = next(iter(syndrome_map.values()))
    n_data = len(first_val)

    syndrome_reg = QuantumRegister(n_syndrome, name='syn')
    data_reg = QuantumRegister(n_data, name='data')
    circuit = QuantumCircuit(syndrome_reg, data_reg, name="syndrome_decode")

    for syndrome_pattern, error_pattern in syndrome_map.items():
        syndrome_bits = syndrome_pattern.zfill(n_syndrome)
        error_bits = error_pattern.zfill(n_data)

        inverted_controls = []
        controls = []
        for idx, bit in enumerate(syndrome_bits):
            if bit == '0':
                circuit.x(syndrome_reg[idx])
                inverted_controls.append(idx)
            controls.append(syndrome_reg[idx])

        for data_idx, bit in enumerate(error_bits):
            if bit == '1':
                if len(controls) == 1:
                    circuit.cx(controls[0], data_reg[data_idx])
                else:
                    circuit.mcx(controls, data_reg[data_idx])
        for idx in inverted_controls:
            circuit.x(syndrome_reg[idx])
        circuit.barrier()
    return circuit

def build_decoding_circuits(m, n, B, ell):
    errors = np.array([
        np.array([1 if i in pos else 0 for i in range(n)])
        for r in range(ell + 1)
        for pos in combinations(range(n), r)
    ])[:, ::-1]

    if m > n:
        errors = np.pad(errors, ((0, 0), (0, m - n)), mode='constant')

    syndromes = (B.T @ errors.T % 2).T
    syndrome_map = {
        "".join(map(str, synd)): "".join(map(str, err))
        for synd, err in zip(syndromes, errors)
    }

    if n > m:
        gje_circuit = QuantumCircuit(n, name="GJE")
        gje_circuit.append(GJEGate(B.T), list(range(n)))
    else:
        gje_circuit = QuantumCircuit(m, name="GJE")
        gje_circuit.append(GJEGate(B.T), list(range(n)))

    lookuptable_circuit = syndrome_decode_lookuptable_gate(syndrome_map)
    return gje_circuit, lookuptable_circuit

def build_complete_dqi_circuit(init_circ, dicke_circ, phase_circ, B_circ, decoding_circ, m, n, use_lookup=False):
    """
    Build the full DQI circuit by composing the initialization, Dicke state preparation,
    phase encoding, constraint encoding, decoding, and Hadamard transform on the syndrome register.
    """
    full_circ = QuantumCircuit(m + n, name=f"Complete DQI Circuit ({'Lookup Table' if use_lookup else 'GJE'})")
    # Compose initialization stages (operate on first m qubits)
    full_circ.compose(init_circ, qubits=list(range(m)), inplace=True)
    full_circ.compose(dicke_circ, qubits=list(range(m)), inplace=True)
    full_circ.compose(phase_circ, qubits=list(range(m)), inplace=True)
    # Compose constraint encoding which acts on m+n qubits
    full_circ.compose(B_circ, qubits=list(range(m + n)), inplace=True)
    # Compose decoding stage on full register
    full_circ.compose(decoding_circ, qubits=list(range(m + n)) if use_lookup else list(range(m if n <= m else n)), inplace=True)
    # Build and compose the Hadamard transform on the syndrome register (last n qubits)
    hadamard_circ = QuantumCircuit(n, name="Hadamard Transform")
    for i in range(n):
        hadamard_circ.h(i)
    full_circ.compose(hadamard_circ, qubits=list(range(m, m + n)), inplace=True)
    return full_circ

def benchmark_gate_info_by_matrix_size(instances):
    benchmark_list = []
    for (num_nodes, num_edges, p, r, ell) in instances:
        entry = {
            "instance": f"{num_nodes}x{num_edges}",
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "p": p,
            "r": r,
            "ell": ell
        }
        result = find_graph_with_target_max_solutions(num_nodes, num_edges, None, 666, 1000)
        if result is None or result[0] is None:
            entry["error"] = "Circuit build failed."
            benchmark_list.append(entry)
            continue
        G, best_assignments, max_satisfied, _ = result
        B, v = get_max_xorsat_matrix(G, v=None)
        m = len(B)
        n = len(B[0])
        w, _ = get_optimal_w(m, n, ell, p, r)
        
        qreg_k = QuantumRegister(m, name='k')
        qreg_y = QuantumRegister(m, name='y')
        qreg_s = QuantumRegister(n, name='s')
        init_circ = QuantumCircuit(qreg_k)
        dicke_circ = QuantumCircuit(qreg_y)
        phase_circ = QuantumCircuit(qreg_y)
        B_circ = QuantumCircuit(qreg_y, qreg_s)
        
        init_circ.append(WeightedUnaryEncoding(m, w), list(range(m)))
        max_errors = int(np.nonzero(w)[0][-1]) if np.any(w) else 0
        dicke_circ.append(UnkStatePreparation(m, max_errors).to_gate(), list(range(m)))
        for i in range(len(v)):
            if v[i] == 1:
                phase_circ.z(i)
        for i in range(n):
            for j in range(m):
                if B.T[i][j] == 1:
                    B_circ.cx(j, m + i)
        
        gje_circuit, lookup_circuit = build_decoding_circuits(m, n, B, ell)
        
        complete_gje = build_complete_dqi_circuit(init_circ, dicke_circ, phase_circ, B_circ, gje_circuit, m, n)
        complete_lookup = build_complete_dqi_circuit(init_circ, dicke_circ, phase_circ, B_circ, lookup_circuit, m, n, use_lookup=True)
        
        entry["Initialization Circuit"] = get_transpiled_gate_info(init_circ, "Initialization Circuit")
        entry["Dicke State Circuit"] = get_transpiled_gate_info(dicke_circ, "Dicke State Circuit")
        entry["Phase Flip Circuit"] = get_transpiled_gate_info(phase_circ, "Phase Flip Circuit")
        entry["Constraint Encoding Circuit"] = get_transpiled_gate_info(B_circ, "Constraint Encoding Circuit")
        
        entry["Decoding"] = {
            "GJE": get_transpiled_gate_info(gje_circuit, "Decoding GJE"),
            "Lookup Table": get_transpiled_gate_info(lookup_circuit, "Decoding Lookup Table")
        }
        
        entry["Complete DQI Circuit (GJE)"] = get_transpiled_gate_info(complete_gje, "Complete DQI Circuit (GJE)")
        entry["Complete DQI Circuit (Lookup Table)"] = get_transpiled_gate_info(complete_lookup, "Complete DQI Circuit (Lookup Table)")
        
        # Build Hadamard transform circuit separately for benchmarking its details
        hadamard_circ = QuantumCircuit(n, name="Hadamard Transform")
        for i in range(n):
            hadamard_circ.h(i)
        entry["Hadamard Transform"] = get_transpiled_gate_info(hadamard_circ, "Hadamard Transform")
        
        benchmark_list.append(entry)
    return benchmark_list

def main():
    report = {}
    instances1 = [(i, i, 2, 1, 2) for i in range(5, 10)]
    report["benchmark"] = benchmark_gate_info_by_matrix_size(instances1)
    with open("info_fix_ell.json", "w") as f:
        json.dump(report, f, indent=4)
    print("Benchmark report saved to 'info_fix_ell.json'.")

    instances2 = [(12, 12, 2, 1, i) for i in range(2, 8)]
    report["benchmark"] = benchmark_gate_info_by_matrix_size(instances2)
    with open("info_fix_B.json", "w") as f:
        json.dump(report, f, indent=4)
    print("Benchmark report saved to 'info_fix_B.json'.")

if __name__ == '__main__':
    main()
