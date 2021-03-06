import numpy
import matplotlib.pyplot as plt
import sys

from matplotlib.ticker import FormatStrFormatter

from plotUtils.plt_utils import save_multiple_formats

__author__ = 'giulio'

"""This script plots train and validation losses for models trained with different number of hidden units"""

# saved file containing the statistics fo the model to compare
stats_files = ['/home/giulio/MuseCollection/0/stats.npz', '/home/giulio/MuseCollection/1/stats.npz',
               '/home/giulio/MuseCollection/2/stats.npz', '/home/giulio/MuseCollection/3/stats.npz']
# color for each model
colors = ['y', 'm', 'r', 'b']
assert (len(colors) == len(stats_files))
mark_freq = 30

n_models = len(stats_files)
legends = []
markers_x = []
markers_y = []
for i in range(n_models):
    stats_file = stats_files[i]
    npz = numpy.load(stats_file)
    check_freq = npz['settings_check_freq']
    length = npz['length']
    x_values = numpy.arange(length) * check_freq
    n_points = x_values.shape[0]
    dots = numpy.arange(start=0, step=mark_freq, stop=n_points)
    x_values_dotted = x_values[dots]
    valid_loss = npz['validation_loss']
    train_loss = npz['train_loss']
    n_hidden = npz['net_n_hidden'].item()
    plt.plot(x_values_dotted, valid_loss[dots], linestyle='dashed', color=colors[i], linewidth=5)
    plt.plot(x_values, train_loss, linestyle='-', color=colors[i], linewidth=1.5)
    legends.append('{} (validation)'.format(n_hidden))
    legends.append('{} (train)'.format(n_hidden))

    best_valid_point_idx = numpy.argmin(valid_loss[0:6 * 10 ** 5 / check_freq])
    markers_x.append(x_values[best_valid_point_idx])
    markers_y.append(valid_loss[best_valid_point_idx])

for i in range(n_models):
    plt.plot(markers_x[i], markers_y[i], '*', color=colors[i], markersize=20)


plt.yscale('log')
plt.ylim(ymin=4, ymax=13)
plt.yticks([4, 5, 6, 7, 8, 9, 10, 13])
plt.xlim(xmin=0, xmax=6 * 10 ** 5)
plt.legend(legends, shadow=True, fancybox=True)
plt.xlabel('iterations')
plt.ylabel('negative log likelihood')

ax = plt.gca()
ax.xaxis.set_major_formatter(FormatStrFormatter("%.0e"))
formatter = FormatStrFormatter("%.1f")
# formatter.set_useOffset(True)
ax.yaxis.set_major_formatter(formatter)
# ax.xaxis.get_major_formatter().set_useOffset(False)
filename = sys.argv[0]
save_multiple_formats(filename)
plt.show()
