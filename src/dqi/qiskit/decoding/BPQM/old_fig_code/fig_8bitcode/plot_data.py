import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
plt.rcParams.update({
    "text.usetex": True,
    "xtick.labelsize" : 12,
    "ytick.labelsize" : 12,
    "axes.labelsize" : 16})

def plot(filename, inset_xrange, inset_yrange):
    data = np.load(filename+".npz")
    angles = data["arr_0"]
    data_h1 = data["arr_1"] 
    data_h2 = data["arr_2"]
    data_h3 = data["arr_3"]
    data_ideal_classical = data["arr_4"]
    data_ideal_quantum = data["arr_5"]

    fig, ax1 = plt.subplots(figsize=(4,5.5))
    ax1.plot(angles, data_h1, label="BPQM h=1", color=(0.5,0.5,1.0))
    ax1.plot(angles, data_h2, label="BPQM h=2", color=(0.3,0.3,0.6))
    ax1.plot(angles, data_h3, label="BPQM h=3", color=(0.1,0.1,0.2))
    ax1.plot(angles, data_ideal_classical, label="ideal classical decoder", color='red')
    ax1.plot(angles, data_ideal_quantum, label="ideal quantum decoder", color='green')
    #ax1.legend(loc='lower right')
    ax1.set_xlabel(r"Channel parameter $\theta$")
    ax1.set_ylabel('Probability of success')
    ax1.set_xticks([0, 0.25*np.pi, 0.5*np.pi])
    ax1.set_xticklabels(['$0$', '$\pi /2$', '$\pi /4$'])

    ax2 = plt.axes([0,0,1,1])
    ax2.set_axes_locator(InsetPosition(ax1, [0.58,0.3,0.4,0.34]))
    mark_inset(ax1, ax2, loc1=1, loc2=3, fc="none", ec='0.5')
    ax2.plot(angles, data_h1, label="BPQM h=1", color=(0.5,0.5,1.0), marker="*")
    ax2.plot(angles, data_h2, label="BPQM h=2", color=(0.3,0.3,0.6), marker="s")
    ax2.plot(angles, data_h3, label="BPQM h=3", color=(0.1,0.1,0.2), marker="d")
    ax2.plot(angles, data_ideal_classical, label="ideal classical decoder", color='red', marker="+")
    ax2.plot(angles, data_ideal_quantum, label="ideal quantum decoder", color='green', marker="p")
    ax2.set_xlim(inset_xrange)
    ax2.set_ylim(inset_yrange)
    ax2.legend(loc=(-0.45,-0.85))

    fig.tight_layout()
    #plt.show()
    plt.savefig(filename+".pdf")


plot("data_x0", inset_xrange=[0.45*0.5*np.pi, 0.55*0.5*np.pi], inset_yrange=[0.87, 0.95])
plot("data_x4", inset_xrange=[0.45*0.5*np.pi, 0.55*0.5*np.pi], inset_yrange=[0.86, 0.95])
plot("data_codeword", inset_xrange=[0.45*0.5*np.pi, 0.55*0.5*np.pi], inset_yrange=[0.7, 0.9])
