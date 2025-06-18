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

ref = decode_bpqm(code, theta, VarNodeCloner(theta), 3, 'bit', bit=0, only_zero_codeword=True, debug=DEBUG)
opt_classical = decode_bit_optimal_classical(code, theta, 0)
opt_quantum = decode_bit_optimal_quantum(code, theta, 0)
x = np.linspace(0.1*theta, 1.9*theta, 21)
vals = list()
for theta_marked in x:
    print(theta_marked)
    vals.append( decode_bpqm(code, theta, OptimalCloner(theta, theta_marked), 3, 'bit', bit=0, only_zero_codeword=False, debug=DEBUG) )

plt.figure()
plt.plot([x[0], x[-1]], [ref, ref], label="variable node cloner")
plt.plot(x, vals, label="optimal cloner")
plt.plot([theta, theta], [np.min(vals), np.max(vals)], label="channel output angle", color='black')
plt.plot([x[0], x[-1]], [opt_classical, opt_classical], label="optimal classical decoder", color='red')
plt.plot([x[0], x[-1]], [opt_quantum, opt_quantum], label="optimal quantum decoder", color='green')
plt.legend(loc='best')
plt.show()


np.savez('data.npz', theta, ref, opt_classical, opt_quantum, x, vals)
