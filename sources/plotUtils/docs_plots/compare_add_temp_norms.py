import sys

import matplotlib.pyplot as plt
import numpy

from plotUtils.plt_utils import save_multiple_formats

stats_file = ['/home/giulio/RNNs/models/train_run/temporal_order, min_length: 100_13/stats.npz',
              '/home/giulio/RNNs/models/add_run/add_task, min_length: 100_13/stats.npz']

labels = ['temp', 'add']
n_files = len(stats_file)
iteration = 5

fig, (ax_r1, ax_r2) = plt.subplots(2, n_files, sharex='col', sharey='row')
x_max_length = 220

plt.rcParams.update({'font.size': 18})

for i in range(n_files):
    npz = numpy.load(stats_file[i])
    label = labels[i]
    y = npz['obj_separate_norms'][iteration]['full_grad']
    ax_r1[i].bar(range(len(y)), y)
    ax_r1[i].legend(['temporal grads norm ({})'.format(label)], shadow=True, fancybox=True)
    ax_r1[i].set_yscale('log')
    # axarr[i].set_xlabel('t')
    # axarr[i].set_ylabel('temporal_grad_norm')
    temporal_dots = npz['obj_grad_temporal_cos'][iteration]
    print(len(temporal_dots))
    ax_r2[i].bar(range(len(temporal_dots)), temporal_dots, color='r')
    ax_r2[i].legend(['temporal cos ({})'.format(label)], shadow=True, fancybox=True)
    ax_r1[i].set_xlim([0, len(y)])
    ax_r2[i].set_xlim([0, len(y)])
    ax_r1[i].set_ylim(ymin=1e-4, ymax=1e1)
    ax_r2[i].set_ylim(ymin=-0.2, ymax=1)
    # ax_r2[i].set_ylim(ymin=1e-2)

filename = sys.argv[0]
save_multiple_formats(filename)
plt.tight_layout(pad=0, w_pad=0.3, h_pad=1)
plt.tight_layout()
plt.show()
