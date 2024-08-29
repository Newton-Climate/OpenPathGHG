import matplotlib.pyplot as plt
import numpy as np
import time
try:
    range1 = np.random.rand(10)
    range2 = np.random.rand(10)
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax2.set_xlim(0, 500)
    graph1 = ax1.plot(range1, range2)[0]
    graph2 = ax2.plot(np.arange(len(range2)), range2)[0]
    # plt.draw()
    plt.show()
    plt.pause(0.01)
    while True:
        range1 = np.append(range1, np.random.rand(1))
        range2 = np.append(range2, np.random.rand(1))
        # range1 = range1[-10:]
        # range2 = range2[-10:]
        time.sleep(0.1)
        # ax1.clear()
        # ax1.plot(range1, range2)
        # ax2.clear()
        # ax2.plot(range2)
        graph1.set_data(range1, range2)
        graph2.set_data(np.arange(len(range2)), np.array(range2))
        # plt.draw()
        plt.pause(0.01)

except KeyboardInterrupt as k:
    plt.ioff()