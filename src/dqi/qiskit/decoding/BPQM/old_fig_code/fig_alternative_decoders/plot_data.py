import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
data = np.load("data.npz")
angles = data["arr_0"]
data_h1 = data["arr_1"] 
data_h2 = data["arr_2"]
data_h3 = data["arr_3"]
data_ideal_classical = data["arr_4"]
data_ideal_quantum = data["arr_5"]
data_strat1 = data["arr_6"]
data_strat2 = data["arr_7"]
data_strat3 = data["arr_8"]

fig, ax1 = plt.subplots(figsize=(9, 6))
#plt.plot(angles, data_h1, label="BPQM h=1", color=(0.5,0.5,1.0))
ax1.plot(angles, data_h2, label="BPQM h=2", color=(0.3,0.3,0.7), marker="s")
ax1.plot(angles, data_h3, label="BPQM h=3", color=(0.1,0.1,0.4), marker="+")
ax1.plot(angles, data_strat3, label="Strategy 3", color=(0.5,1.0,0.5), marker="*")
ax1.plot(angles, data_strat2, label="Strategy 2", color=(0.3,0.7,0.3), marker='d')
ax1.plot(angles, data_strat1, label="Strategy 1", color=(0.1,0.4,0.1), marker="x")
#plt.plot(angles, data_ideal_classical, label="ideal classical decoder", color='red')
#plt.plot(angles, data_ideal_quantum, label="ideal quantum decoder", color='green')
ax1.legend(loc='best')
ax1.set_xlabel(r"Channel parameter $\theta$")
ax1.set_xticks([0., 0.25*0.5*np.pi, 0.5*0.5*np.pi, 0.75*0.5*np.pi, 0.5*np.pi])
ax1.set_xticklabels(['$0$', '$\pi /8$', '$\pi /4$', '$3\pi /8$', '$\pi /2$'])
ax1.set_ylabel('Probability of successfully decoding $X_1$')


# inset plot
ax2 = plt.axes([0,0,1,1])
ax2.set_axes_locator(InsetPosition(ax1, [0.68,0.5,0.3,0.4]))
mark_inset(ax1, ax2, loc1=2, loc2=3, fc="none", ec='0.5')
#plt.plot(angles, data_h1, label="BPQM h=1", color=(0.5,0.5,1.0))
ax2.plot(angles, data_h2, label="BPQM h=2", color=(0.3,0.3,0.7), marker="s")
ax2.plot(angles, data_h3, label="BPQM h=3", color=(0.1,0.1,0.4), marker="+")
ax2.plot(angles, data_strat3, label="Strategy 3", color=(0.5,1.0,0.5), marker="*")
ax2.plot(angles, data_strat2, label="Strategy 2", color=(0.3,0.7,0.3), marker='d')
ax2.plot(angles, data_strat1, label="Strategy 1", color=(0.1,0.4,0.1), marker="x")
#plt.plot(angles, data_ideal_classical, label="ideal classical decoder", color='red')
#plt.plot(angles, data_ideal_quantum, label="ideal quantum decoder", color='green')
ax2.set_xlim([0.6, 0.7])
ax2.set_ylim([0.86, 0.90])

#plt.show()
plt.savefig("alternative_decoders.pdf")
