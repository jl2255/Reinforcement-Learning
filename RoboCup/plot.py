from matplotlib import pyplot as plt
import numpy as np
plt.rcParams['agg.path.chunksize'] = 10000


def plot(values, title, figName):
    diff = abs(values[:-1] - values[1:])
    axes = plt.gca()
    #ymin, ymax = axes.get_ylim()
    #xmin, xmax = axes.get_xlim()
    axes.set_ylim([0, 0.5])
    axes.set_xlim([0, 1000000])
    #axes.set_xlim([xmin, xmax])
    plt.xlabel("Simulation Iteration")
    plt.ylabel("Q-value Difference")
    plt.suptitle(title)
    plt.savefig(figName)
    plt.plot(diff)
    plt.show()