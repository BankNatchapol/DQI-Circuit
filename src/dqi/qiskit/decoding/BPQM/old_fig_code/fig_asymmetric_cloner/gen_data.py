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

theta = 0.2*np.pi
opt_classical = decode_bit_optimal_classical(code, theta, 0)
opt_quantum = decode_bit_optimal_quantum(code, theta, 0)
ref2 = decode_bpqm(code, theta, VarNodeCloner(theta), 2, 'bit', bit=4, only_zero_codeword=True, debug=DEBUG)
ref3 = decode_bpqm(code, theta, VarNodeCloner(theta), 3, 'bit', bit=4, only_zero_codeword=True, debug=DEBUG)

x = np.linspace(0.0, 1.0, 11)
vals = list()
for frac in x:
    print(frac)
    vals.append( decode_bpqm(code, theta, AsymmetricVarNodeCloner(theta, frac, 4), 3, 'bit', bit=4, only_zero_codeword=True, debug=DEBUG) )

plt.figure()
plt.plot(x, vals, label="asymmetric variable node cloner")
plt.plot([x[0], x[-1]], [ref2, ref2], label="variable node cloner h=2")
plt.plot([x[0], x[-1]], [ref3, ref3], label="variable node cloner h=3")
plt.plot([x[0], x[-1]], [opt_classical, opt_classical], label="optimal classical decoder", color='red')
plt.plot([x[0], x[-1]], [opt_quantum, opt_quantum], label="optimal quantum decoder", color='green')
plt.legend(loc='best')
plt.show()


np.savez('data.npz', opt_classical, opt_quantum, ref2, ref3, x, vals)
