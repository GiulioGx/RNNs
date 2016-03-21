import numpy
import matplotlib.pyplot as plt
import sys

from matplotlib.ticker import FormatStrFormatter

from plotUtils.plt_utils import save_multiple_formats

__author__ = 'giulio'

"""This script plots train and validation losses for models trained with different number of hidden units"""

# saved file containing the statistics fo the model to compare
stats_files = ['/home/giulio/tmpOct/oct1/stats.npz', '/home/giulio/tmpOct/oct2/stats.npz',
               '/home/giulio/tmpOct/oct3/stats.npz']
# color for each model
colors = ['y', 'm', 'r']
assert (len(colors) == len(stats_files))
n_models = len(stats_files)
legends = ['anti-gradient', 'simplex', 'simplex with conditional switching']


for i in range(n_models):
    stats_file = stats_files[i]
    npz = numpy.load(stats_file)
    check_freq = npz['settings_check_freq']
    length = npz['length']
    x_values = numpy.arange(length) * check_freq
    n_points = x_values.shape[0]
    valid_loss = npz['validation_loss']
    plt.plot(x_values, valid_loss, 'o', color=colors[i])
    legends.append(legends[i])

plt.rcParams.update({'font.size': 24})
plt.yscale('log')
# plt.ylim(ymin=4, ymax=13)
# plt.yticks([4, 5, 6, 7, 8, 9, 10, 13])
# plt.xlim(xmin=0, xmax=6 * 10 ** 5)
plt.legend(legends, shadow=True, fancybox=True)
plt.xlabel('iterations')
plt.ylabel('validation loss')

ax = plt.gca()
# ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
# ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

filename = sys.argv[0]
save_multiple_formats(filename)
plt.show()
