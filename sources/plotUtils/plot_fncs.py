import matplotlib.pyplot as plt


__author__ = 'giulio'


def plot_norms(norms):

    fig, ax = plt.subplots(figsize=(20, 30))
    x = range(len(norms))
    ax.bar(x, norms)
    return fig
