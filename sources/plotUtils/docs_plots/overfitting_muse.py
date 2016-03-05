import numpy
import matplotlib.pyplot as plt
import sys

from plotUtils.plt_utils import save_multiple_formats

__author__ = 'giulio'

"""This script plots train and validation losses for models trained with different number of hidden units"""

# saved file containing the statistics fo the model to compare
stats_files = []
# color for each model
colors = []
assert(len(colors)==len(stats_files))

n_models = len(stats_files)
legends = []
for i in range(n_models):
    stats_file = stats_files[i]
    npz = numpy.load(stats_file)
    check_freq = npz['settings_check_freq']
    length = npz['length']
    x_values = numpy.arange(length) * check_freq
    valid_loss = npz['validation_loss']
    train_loss = npz['train_loss']
    n_hidden = npz['net_n_hidden'].item()
    plt.plot(x_values, valid_loss, linestyle='--', color=colors[i])
    plt.plot(x_values, train_loss, linestyle='-', color=colors[i])
    legends.append(n_hidden)

plt.legend(legends, shadow=True, fancybox=True)
plt.set_yscale('log')
plt.set_xlabel('iterations')
plt.set_ylabel('negative log likelihood')

filename = sys.argv[0]
save_multiple_formats(filename)
plt.show()