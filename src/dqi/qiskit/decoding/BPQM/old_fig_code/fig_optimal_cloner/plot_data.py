import numpy as np
from matplotlib import pyplot as plt

plt.rcParams.update({"text.usetex": True})

data = np.load('data.npz')
theta = data['arr_0']
ref = data['arr_1']
opt_classical = data['arr_2']
opt_quantum = data['arr_3']
x = data['arr_4']
vals = data['arr_5']

plt.figure(figsize=(6, 4))
plt.plot([0., np.pi], [ref, ref], '--', label="ENU cloner")
plt.plot(x, vals, label="optimal cloner", color='purple')
plt.plot([0., np.pi], [opt_classical, opt_classical], '--', label="optimal classical decoder", color='red')
plt.plot([0., np.pi], [opt_quantum, opt_quantum], '--', label="optimal quantum decoder", color='green')

plt.plot([theta, theta], [-1., 2.], color='black')
plt.plot([np.arccos(np.sqrt(np.cos(theta))), np.arccos(np.sqrt(np.cos(theta)))], [-1., 2.], color='black')
#plt.text(theta, 0.83, " $0.2\pi$")
#plt.text(np.arccos(np.sqrt(np.cos(theta))), 0.83, "$arccos\sqrt{cos(0.2\pi)}$ ", horizontalalignment='right')


plt.xlim(0., 1.25) 
plt.ylim(0.8, 0.9)
plt.xlabel(r"BPQM channel parameter for approximate clones $\theta'$")
plt.xticks(ticks=[0., np.arccos(np.sqrt(np.cos(theta))), theta, 0.75*0.5*np.pi],
           labels=[r"$0$", r"\small$\arccos\sqrt{cos(0.2\pi)}$", r"$0.2\pi$", "$3\pi /8$"])
plt.ylabel('Probability of successfully decoding $X_1$')
plt.legend(loc='best')

plt.tight_layout()
#plt.show()
plt.savefig('optimal_cloner.pdf')


