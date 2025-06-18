from typing import Optional, Sequence, Tuple, List

import numpy as np
from numpy.typing import NDArray
import qiskit
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.library import SaveProbabilitiesDict

# Use the actual Qiskit QuantumCircuit
QuantumCircuit = qiskit.QuantumCircuit

try:
    from .bpqm import tree_bpqm
    from .cloner import Cloner
    from .linearcode import LinearCode
except ImportError:
    from bpqm import tree_bpqm
    from cloner import Cloner
    from linearcode import LinearCode


def TP(exprs: Sequence[NDArray]) -> NDArray:
    """Return the Kronecker product of all matrices in ``exprs``."""
    out = exprs[0]
    for mat in exprs[1:]:
        out = np.kron(out, mat)
    return out

def decode_bpqm(
    code: LinearCode,
    theta: float,
    cloner: Cloner,
    height: int,
    mode: str,
    bit: Optional[int] = None,
    order: Optional[Sequence[int]] = None,
    only_zero_codeword: bool = True,
    debug: bool = False,
) -> float:
    """Evaluate BPQM success probability for a bit or full codeword.

    Parameters
    ----------
    code : LinearCode
        Code to decode.
    theta : float
        Channel parameter.
    cloner : Cloner
        Cloner used for unrolled variable nodes.
    height : int
        Unrolling depth for the computation tree.
    mode : {"bit", "codeword"}
        Decode a single bit or the entire codeword.
    bit : int, optional
        Index of the bit to decode when ``mode='bit'``.
    order : Sequence[int], optional
        Order in which bits are decoded. Defaults to ``range(code.n)``.
    only_zero_codeword : bool, optional
        If ``True`` only the all-zero codeword is simulated.
    debug : bool, optional
        If ``True`` print intermediate circuit information.

    Returns
    -------
    float
        Average success probability over the simulated codewords.
    """
    assert mode in ['bit','codeword'], "mode should be 'bit' or 'codeword'."
    if mode=='bit':
        assert bit!=None, "bit shouldn't be None when choosing mode'bit'."
        order=[bit]

    if order == None:
        order = list(range(code.n))

    # 1) build computation graphs
    cgraphs = [code.get_computation_graph(f"x{b}", height) for b in order]

    # 2) determine qubit counts
    n_data_qubits = max(sum(occ.values()) for _,occ,_ in cgraphs)
    n_data_qubits = max(n_data_qubits, code.n)
    n_qubits = n_data_qubits + len(order)-1

    # 3) generate main circuit
    meas_idx = 0
    qc = QuantumCircuit(n_qubits)
    for i,(graph,occ,root) in enumerate(cgraphs):
        # qubit mapping
        leaves = [n for n in graph.nodes() if graph.nodes[n]["type"]=="output"]
        leaves = sorted(leaves, key=lambda s:int(s.split("_")[1]))
        qm = {f"y{j}_0":j for j in range(code.n)}
        idx=code.n
        for node in leaves:
            if int(node.split("_")[1])>0:
                qm[node]=idx; idx+=1

        # annotate graph
        cloner.mark_angles(graph, occ)
        for node in leaves:
            graph.nodes[node]["qubit_idx"]=qm[node]

        # build BPQM + cloner
        qc_bpqm = QuantumCircuit(n_qubits)
        meas_idx, angles = tree_bpqm(graph, qc_bpqm, root=root)
        qc_cloner = cloner.generate_cloner_circuit(graph, occ, qm, n_qubits)

        # append & uncompute with compose
        qc.compose(qc_cloner, inplace=True); qc.barrier()
        qc.compose(qc_bpqm,   inplace=True); qc.barrier()
        if i < len(order)-1:
            qc.h(meas_idx); qc.cx(meas_idx, n_data_qubits+i); qc.h(meas_idx); qc.barrier()
            qc.compose(qc_bpqm.inverse(),   inplace=True); qc.barrier()
            qc.compose(qc_cloner.inverse(), inplace=True); qc.barrier()
        else:
            qc.h(meas_idx)

    # snapshot as dict
    cw_qubits = list(range(n_data_qubits, n_data_qubits+len(order)-1)) + [meas_idx]
    qc.append(SaveProbabilitiesDict(len(cw_qubits), label='prob'), cw_qubits)
    # qc.save_probabilities_dict(label='prob', qubits=cw_qubits)

    # simulate
    backend = AerSimulator(method='statevector')
    codewords = [[0]*code.n] if only_zero_codeword else code.get_codewords()
    prob=0.
    for cw in codewords:
        qc_init = QuantumCircuit(n_qubits)
        plus  = np.array([np.cos(0.5*theta),  np.sin(0.5*theta)])
        minus = np.array([np.cos(0.5*theta), -np.sin(0.5*theta)])
        for j,v in enumerate(cw):
            state = (plus if v == 0 else minus).tolist()           # ← convert here
            qc_init.initialize(state, [j])

        combined = qc_init.compose(qc)
        assert combined is not None, "Unexpected None for combined"

        full_qc   = transpile(combined, backend)
        result    = backend.run(full_qc).result()

        probs = result.data()['prob']
        key   = int("".join(str(cw[i]) for i in reversed(order)),2)
        prob += probs.get(key,0.0)/len(codewords)

    return prob

def create_init_qc(
    code: LinearCode,
    theta: float,
    codeword: Optional[NDArray[np.int_]] = None,
    prior: Optional[float] = None
) -> QuantumCircuit:
    """
    Prepare an initialization circuit for data qubits.

    If `prior` is None, initializes each qubit to |+⟩ or |−⟩ states at angle θ
    according to `codeword`. Otherwise prepares a uniform prior state
    with amplitude mix determined by `prior` and θ.

    Parameters
    ----------
    code : LinearCode
        Linear code defining number of qubits (code.n).
    theta : float
        Rotation angle for RY gates.
    codeword : Optional[NDArray[np.int_]]
        Binary array of length code.n; required if prior is None.
    prior : Optional[float]
        Probability weight for |0⟩ in the prior mixture.

    Returns
    -------
    QuantumCircuit
        Circuit initializing `code.n` qubits.
    """
    n = code.n
    qc = QuantumCircuit(n)

    if prior is None:
        if codeword is None:
            raise ValueError("codeword must be provided when prior is None")
        for j, bit in enumerate(codeword):
            angle = theta if bit == 0 else -theta
            qc.ry(angle, j)
    else:
        # Build mixture amplitudes [a, b]
        mix = np.array([
            prior * np.cos(theta / 2) + (1 - prior) * np.cos(theta / 2),
            prior * np.sin(theta / 2) - (1 - prior) * np.sin(theta / 2)
        ], dtype=float)
        mix /= np.linalg.norm(mix)
        theta_mix = 2 * np.arctan2(mix[1], mix[0])
        for j in range(n):
            qc.ry(theta_mix, j)
            if mix[1] < 0:
                qc.z(j)

    return qc


def decode_single_codeword(
    qc_init: QuantumCircuit,
    code: LinearCode,
    cloner: Cloner,
    height: int,
    shots: int = 512,
    debug: bool = False,
    run_simulation: bool = True
) -> Tuple[Optional[np.ndarray], List[int], QuantumCircuit]:
    """Decode a single codeword using the BPQM circuit.

    Parameters
    ----------
    qc_init : QuantumCircuit
        Circuit produced by :func:`create_init_qc` that prepares the inputs.
    code : LinearCode
        Linear-code instance describing the factor graph.
    cloner : Cloner
        Cloner used to approximate variable-node copies.
    height : int
        Unrolling depth for the BPQM computation tree.
    shots : int, optional
        Number of measurement shots (default ``512``).
    debug : bool, optional
        If ``True`` print measurement counts and syndromes.
    run_simulation : bool, optional
        If ``False`` only the circuit is returned.

    Returns
    -------
    decoded_bits : np.ndarray or None
        Best guess for the codeword, or ``None`` when ``run_simulation`` is ``False``.
    decoded_qubits : List[int]
        Indices of the measured qubits in order.
    qc_decode : QuantumCircuit
        The constructed BPQM circuit (without ``qc_init``).
    """
    order = list(range(code.n))

    # 1) Build all BPQM subcircuits
    cgraphs = [code.get_computation_graph(f"x{b}", height) for b in order]
    n_data_qubits = max(sum(occ.values()) for _, occ, _ in cgraphs)
    n_data_qubits = max(n_data_qubits, code.n)
    n_total_qubits = n_data_qubits + len(order) - 1

    qc_decode = QuantumCircuit(n_total_qubits)
    meas_idx = 0

    for i, (graph, occ, root) in enumerate(cgraphs):
        # map each "y" (output) node to a physical qubit
        leaves = sorted(
            [n for n in graph.nodes() if graph.nodes[n]["type"] == "output"],
            key=lambda s: int(s.split("_")[1])
        )
        qubit_map = {f"y{j}_0": j for j in range(code.n)}
        idx = code.n
        for leaf in leaves:
            if int(leaf.split("_")[1]) > 0:
                qubit_map[leaf] = idx
                idx += 1

        # mark angles & build the BPQM + cloner pieces
        cloner.mark_angles(graph, occ)
        for leaf in leaves:
            graph.nodes[leaf]["qubit_idx"] = qubit_map[leaf]
        qc_bpqm, _ = QuantumCircuit(n_total_qubits), None
        meas_idx, _ = tree_bpqm(graph, qc_bpqm, root=root)
        qc_cloner = cloner.generate_cloner_circuit(
            graph, occ, qubit_map, n_total_qubits
        )

        qc_decode.compose(qc_cloner, inplace=True)
        qc_decode.barrier()
        qc_decode.compose(qc_bpqm, inplace=True)
        qc_decode.barrier()

        if i < len(order) - 1:
            qc_decode.h(meas_idx)
            qc_decode.cx(meas_idx, n_data_qubits + i)
            qc_decode.h(meas_idx)
            qc_decode.barrier()
            qc_decode.compose(qc_bpqm.inverse(), inplace=True)
            qc_decode.barrier()
            qc_decode.compose(qc_cloner.inverse(), inplace=True)
            qc_decode.barrier()
        else:
            qc_decode.h(meas_idx)

    decoded_qubits = list(range(n_data_qubits,
                                n_data_qubits + len(order) - 1)) + [meas_idx]

    # If not running simulation, return placeholders
    if not run_simulation:
        return None, decoded_qubits, qc_decode

    # 2) Compose init + decode + measurements onto a full-width circuit
    full_qc = QuantumCircuit(n_total_qubits, len(order))
    full_qc.compose(
        qc_init,
        qubits=list(range(qc_init.num_qubits)),
        inplace=True
    )
    full_qc.compose(qc_decode, inplace=True)
    for idx, qb in enumerate(decoded_qubits):
        full_qc.measure(qb, idx)

    # 3) Run and post-process
    backend = AerSimulator()
    job     = backend.run(transpile(full_qc, backend), shots=shots)
    result  = job.result().get_counts()

    reversed_counts = {bits[::-1]: cnt for bits, cnt in result.items()}
    sorted_counts   = dict(sorted(
        reversed_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ))

    if debug:
        print("Counts:")
        for bits, cnt in sorted_counts.items():
            syn_vec = (np.array([int(b) for b in bits]) @ code.H.T) % 2
            print(f"  {bits} → {cnt} : syndrome {syn_vec}")

    best = next(iter(sorted_counts))
    decoded_bits = np.array([int(b) for b in best], dtype=int)
    return decoded_bits, decoded_qubits, qc_decode

def decode_single_syndrome(
    syndrome_qc: QuantumCircuit,
    code: LinearCode,
    theta: float,
    prior: Optional[float] = None,
    height: int = 2,
    shots: int = 512,
    debug: bool = False,
    run_simulation: bool = True
) -> Tuple[Optional[np.ndarray], Optional[List[int]], QuantumCircuit]:
    """
    Perform syndrome decoding via BPQM without a separate cloner stage.

    Parameters
    ----------
    syndrome_qc : QuantumCircuit
        Circuit that prepares syndrome ancilla bits.
    code : LinearCode
        Linear code describing parity checks (provides `n` and `H`).
    theta : float
        Rotation angle for initializing data qubits.
    prior : Optional[float], default=None
        Prior probability for the |0> state (uniform initialization if provided).
    height : int, default=2
        Depth of unrolling for the BPQM factor-tree.
    shots : int, default=512
        Number of measurement shots when running the simulation.
    debug : bool, default=False
        If True, print measurement counts and syndrome diagnostics.
    run_simulation : bool, default=True
        If False, return the decoding circuit without execution.

    Returns
    -------
    decoded_bits : Optional[np.ndarray]
        The most likely decoded bitstring, or None if not simulated.
    decoded_qubits : Optional[List[int]]
        List of classical register indices for the decoded bits, or None.
    qc_decode : QuantumCircuit
        The constructed BPQM decoding circuit (excluding initialization and syndrome prep).
    """
    # Determine ordering and build computation graphs
    order = list(range(code.n))
    computation_graphs = [
        code.get_computation_graph(f"x{b}", height, syndrome_mode=True)
        for b in order
    ]

    # Determine qubit counts
    n_ancilla = code.H.shape[0]
    max_data = max(sum(occ.values()) for _, occ, _ in computation_graphs)
    n_data = max(max_data, code.n)
    total_qubits = n_ancilla + n_data + len(order)

    # Initialize decode circuit and data preparation
    qc_decode = QuantumCircuit(total_qubits)
    
    # Create a temporary code for initialization
    temp_code = code.__class__(None, np.zeros((0, n_data), int))
    temp_code.n = n_data
    
    # If prior is None, use all-zero codeword
    if prior is None:
        codeword = np.zeros(n_data, dtype=int)
    else:
        codeword = None
    
    data_init = create_init_qc(
        code=temp_code,
        theta=theta,
        codeword=codeword,
        prior=prior
    )
    qc_decode.compose(
        data_init,
        qubits=list(range(n_ancilla, n_ancilla + n_data)),
        inplace=True
    )

    # Build and apply BPQM for each logical qubit
    for idx, (graph, occ, root) in enumerate(computation_graphs):
        # Map output nodes to qubit indices
        leaves = [n for n, d in graph.nodes(data=True) if d.get("type") == "output"]
        qubit_map = {f"y{j}_0": j for j in range(code.n)}
        next_idx = code.n
        for leaf in leaves:
            level = int(leaf.split("_")[1])
            if level > 0:
                qubit_map[leaf] = next_idx
                next_idx += 1

        # Annotate output angles and qubit indices
        for leaf in leaves:
            count = occ[leaf.split("_")[0].replace("y", "x")]
            angle = np.arccos(np.cos(theta) ** (1.0 / count))
            graph.nodes[leaf]["angle"] = [(angle, {})]
            graph.nodes[leaf]["qubit_idx"] = qubit_map[leaf]

        # Remove unused check nodes
        to_remove = [n for n, d in graph.nodes(data=True)
                     if d.get("type") == "check" and graph.out_degree(n) == 0]
        graph.remove_nodes_from(to_remove)

        # Construct BPQM subcircuit
        qc_bpqm = QuantumCircuit(total_qubits)
        meas_idx, _ = tree_bpqm(graph, qc_bpqm, root=root, offset=n_ancilla)
        qc_decode.compose(qc_bpqm, inplace=True)
        qc_decode.barrier()

        # Entangling measurement and uncompute
        qc_decode.h(meas_idx)
        qc_decode.cx(meas_idx, n_ancilla + n_data + idx)
        qc_decode.h(meas_idx)
        qc_decode.barrier()
        qc_decode.compose(qc_bpqm.inverse(), inplace=True)
        qc_decode.barrier()

    # Uncompute data initialization
    qc_decode.compose(
        data_init.inverse(),
        qubits=list(range(n_ancilla, n_ancilla + n_data)),
        inplace=True
    )

    if not run_simulation:
        return None, None, qc_decode

    # Prepare full circuit with syndrome prep and measurement
    decoded_qubits = list(range(n_ancilla + n_data, total_qubits))
    full_qc = QuantumCircuit(total_qubits, len(order))
    full_qc.compose(syndrome_qc, qubits=list(range(n_ancilla)), inplace=True)
    full_qc.compose(qc_decode, inplace=True)
    for i, qb in enumerate(decoded_qubits):
        full_qc.measure(qb, i)

    # Execute on simulator
    backend = AerSimulator()
    compiled = transpile(full_qc, backend)
    job = backend.run(compiled, shots=shots)
    counts = job.result().get_counts()

    # Process results
    flipped = {bits[::-1]: cnt for bits, cnt in counts.items()}
    sorted_counts = dict(sorted(flipped.items(), key=lambda x: x[1], reverse=True))

    if debug:
        print("Counts:")
        for bits, cnt in sorted_counts.items():
            syn = (np.array(list(bits), dtype=int) @ code.H.T) % 2
            print(f"  {bits} → {cnt} : syndrome {syn}")

    best = next(iter(sorted_counts))
    decoded_bits = np.array([int(b) for b in best], dtype=int)
    return decoded_bits, decoded_qubits, qc_decode
