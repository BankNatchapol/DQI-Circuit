import sys
sys.path.append("..")
from decoders import *

DEBUG = False 


# 8-bit code
G = np.array([[1, 0, 0, 0, 1, 0, 0, 1],
              [0, 1, 0, 0, 1, 1, 0, 0],
              [0, 0, 1, 0, 0, 1, 1, 0],
              [0, 0, 0, 1, 0, 0, 1, 1]])
H = np.array([[1, 1, 0, 0, 1, 0, 0, 0],
              [0, 1, 1, 0, 0, 1, 0, 0],
              [0, 0, 1, 1, 0, 0, 1, 0],
              [1, 0, 0, 1, 0, 0, 0, 1]])
code = LinearCode(G, H)
if DEBUG:
    # plot code factor graph
    nx.draw(code.get_factor_graph(), with_labels=True)
    plt.show()

# parameters for data generation
N = 40
ch_angles = np.linspace(0., 0.5*np.pi, N)[1:]


print("Generating data on x0 decoding...")
data_h1 = list()
data_h2 = list()
data_h3 = list()
data_ideal_classical = list()
data_ideal_quantum = list()
for theta in ch_angles:
    data_h1.append( decode_bpqm(code, theta, VarNodeCloner(theta), 1, 'bit', bit=0, only_zero_codeword=True, debug=DEBUG) )
    data_h2.append( decode_bpqm(code, theta, VarNodeCloner(theta), 2, 'bit', bit=0, only_zero_codeword=True, debug=DEBUG) )
    data_h3.append( decode_bpqm(code, theta, VarNodeCloner(theta), 3, 'bit', bit=0, only_zero_codeword=True, debug=DEBUG) )
    data_ideal_classical.append( decode_bit_optimal_classical(code, theta, 0) )
    data_ideal_quantum.append( decode_bit_optimal_quantum(code,theta, 0) )
np.savez('data_x0.npz', ch_angles, data_h1, data_h2, data_h3, data_ideal_classical, data_ideal_quantum)

print("Generating data on x4 decoding...")
data_h1 = list()
data_h2 = list()
data_h3 = list()
data_ideal_classical = list()
data_ideal_quantum = list()
for theta in ch_angles:
    data_h1.append( decode_bpqm(code, theta, VarNodeCloner(theta), 1, 'bit', bit=4, only_zero_codeword=True, debug=DEBUG) )
    data_h2.append( decode_bpqm(code, theta, VarNodeCloner(theta), 2, 'bit', bit=4, only_zero_codeword=True, debug=DEBUG) )
    data_h3.append( decode_bpqm(code, theta, VarNodeCloner(theta), 3, 'bit', bit=4, only_zero_codeword=True, debug=DEBUG) )
    data_ideal_classical.append( decode_bit_optimal_classical(code, theta, 4) )
    data_ideal_quantum.append( decode_bit_optimal_quantum(code,theta, 4) )
np.savez('data_x4.npz', ch_angles, data_h1, data_h2, data_h3, data_ideal_classical, data_ideal_quantum)

print("Generating data on completete codeword decoding...")
data_h1 = list()
data_h2 = list()
data_h3 = list()
data_ideal_classical = list()
data_ideal_quantum = list()
for theta in ch_angles:
    data_h1.append( decode_bpqm(code, theta, VarNodeCloner(theta), 1, 'codeword', order=[0,1,2,3], only_zero_codeword=True, debug=DEBUG) )
    data_h2.append( decode_bpqm(code, theta, VarNodeCloner(theta), 2, 'codeword', order=[0,1,2,3], only_zero_codeword=True, debug=DEBUG) )
    data_h3.append( decode_bpqm(code, theta, VarNodeCloner(theta), 3, 'codeword', order=[0,1,2,3], only_zero_codeword=True, debug=DEBUG) )
    data_ideal_classical.append( decode_codeword_optimal_classical(code, theta) )
    # I could also use decode_codeword_optimal_quantum here which would solve the distinguishability SDP
    # but the PGM is faster and gives the same result.
    data_ideal_quantum.append( decode_codeword_PGM(code,theta) )
np.savez('data_codeword.npz', ch_angles, data_h1, data_h2, data_h3, data_ideal_classical, data_ideal_quantum)
