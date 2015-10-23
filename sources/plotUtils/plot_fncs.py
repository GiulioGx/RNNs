import matplotlib.pyplot as plt


__author__ = 'giulio'


def plot_norms(norms):

    fig, ax = plt.subplots()
    x = range(len(norms))
    ax.bar(x, norms)
    plt.savefig("grads.svg")
    plt.show()
