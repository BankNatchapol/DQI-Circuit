import numpy as np
from state import *
from matplotlib import pyplot as plt

plt.rcParams.update({
    "text.usetex": True})

#           1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17
codeword = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
angle = 0.2*np.pi

def get_err(B):
    state = State(17, angle, codeword, B)

    state.apply_checknode_op(1, 2)
    state.apply_checknode_op(3, 4)
    state.apply_checknode_op(5, 6)
    state.apply_checknode_op(7, 8)
    state.apply_checknode_op(9, 10)
    state.apply_checknode_op(11, 12)
    state.apply_checknode_op(13, 14)
    state.apply_checknode_op(15, 16)

    state.apply_eqnode_op(1, 3)
    state.apply_eqnode_op(5, 7)
    state.apply_eqnode_op(9, 11)
    state.apply_eqnode_op(13, 15)

    state.apply_checknode_op(1, 5)
    state.apply_checknode_op(9, 13)

    state.apply_eqnode_op(1, 9)

    state.apply_eqnode_op(0, 1)

    approximate = state.measure(0)[0]
    ideal = state.ideal_distinguishability(0)
    return approximate, ideal

def paper_bound(B, n):
    return (4.*np.sqrt(2.)/3.) * np.pi * 2.**( n * (1.5 + np.log(26)/np.log(2.)) - 0.5*B )

xvals = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
yvals = list()
yvals_bound = list()
for B in xvals:
    print(B)
    approximate, yval_ideal = get_err(B)
    yvals.append(approximate)
    yvals_bound.append(paper_bound(B, 17))


plt.figure(figsize=(4,3))
plt.semilogy(xvals, np.abs(yvals-yval_ideal), '--bo', label='simulation')
plt.tight_layout()

plt.gcf().subplots_adjust(left=0.18, bottom=0.17)
plt.xlabel(r"Number of qubits per angle register $B$")
plt.ylabel(r"Decoder suboptimality $\epsilon$")
plt.savefig("17bitcode_discretisation_errors.pdf")

