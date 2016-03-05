import matplotlib.pyplot as plt

__author__ = 'giulio'


def save_multiple_formats(filename: str, *formats: str):
    """save the figure mutiple time in every format specified in 'formats'"""

    if len(formats) == 0:
        formats = ['eps', 'svg', 'pgf']
    for f in formats:
        plt.savefig(filename + '_.' + f, transparent=True, format=f)
