import numpy as np

from qiskit_impl.decoders import decode_bpqm, decode_single_codeword, decode_single_syndrome, create_init_qc
from qiskit_impl.cloner import ExtendedVarNodeCloner
from qiskit_impl.linearcode import LinearCode

from qiskit import QuantumCircuit

H = np.array([
    [1, 0, 0, 1, 0],
    [1, 0, 1, 0, 0],
    [1, 1, 1, 1, 0],
])
# H = np.array([
#     [1, 1, 0, 0, 1, 0, 0, 0],
#     [0, 1, 1, 0, 0, 1, 0, 0],
#     [0, 0, 1, 1, 0, 0, 1, 0],
#     [1, 0, 0, 1, 0, 0, 0, 1],
# ])
theta = 0.2 * np.pi
cloner = ExtendedVarNodeCloner(theta)
code = LinearCode(None, H)

corrupted_codeword = np.array([1, 1, 0, 1, 1])
syndrome = corrupted_codeword @ H.T %2

qc_init = create_init_qc(
    code=code,
    codeword=None,
    theta=theta,
    prior=0.5      # or set a prior like 0.5
)

syndrome_qc = QuantumCircuit(len(syndrome))
for i, s in enumerate(syndrome):
    if s == 1:
        syndrome_qc.x(i)

decoded_bits, decoded_qubits, qc_decode = decode_single_syndrome(
    syndrome_qc=syndrome_qc,
    code=code,
    prior=0.5,
    theta=theta,
    height=2,
    shots=10000,
    debug=True,
    run_simulation=True
)
print("syndrome             : ", syndrome)
print("Corrupted codeword   : ", corrupted_codeword)
print("Decoded bits         : ", decoded_bits)
if decoded_bits is not None:
    print("Decoded bits syndrome: ", decoded_bits @ H.T %2)