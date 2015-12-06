import matplotlib.pyplot as plt
import numpy

__author__ = 'giulio'

modelFile = '/home/giulio/RNNs/models/completed/temporal_order, min_length: 144/model.npz'
#modelFile = '/home/giulio/model_octopus.npz'

# load npz archive
npz = numpy.load(modelFile)


# choose what and how to display it
valid_error = {'label': 'validation_error_curr', 'legend': 'validation error', 'color': 'r', 'scale': 'linear'}
valid_loss = {'label': 'validation_loss', 'legend': 'validation loss', 'color': 'm', 'scale': 'linear'}
dir_norm = {'label': 'dir_norm', 'legend': 'dir norm', 'color': 'm', 'scale': 'log'}
grad_dot = {'label': 'grad_dot', 'legend': 'gradient dot', 'color': 'b', 'scale': 'linear'}
equi_cos = {'label': 'W_rec_equi_cos', 'legend': 'equi_cos', 'color': 'b', 'scale': 'linear'}
rho = {'label': 'rho', 'legend': 'W_rec spetral radius', 'color': 'r', 'scale': 'linear'}


measures = [valid_error, valid_loss, dir_norm, rho]

# compute x-axis points
check_freq = npz['settings_check_freq']
length = npz['length']
x_values = numpy.arange(length) * check_freq

# plot
n_plots = len(measures)
fig, axarr = plt.subplots(n_plots, sharex=True)

for i in range(n_plots):
    measure = measures[i]
    print(measure)
    y_values = npz[measure['label']]
    axarr[i].plot(x_values, y_values, measure['color'])
    axarr[i].legend([measure['legend']], shadow=True, fancybox=True)
    axarr[i].set_yscale(measure['scale'])


# description
elapsed_time = npz['elapsed_time'].item()
n_hidden = npz['net_n_hidden'].item()
n_in = npz['net_n_in'].item()
n_out = npz['net_n_out'].item()
activation_fnc = npz['net_activation_fnc']
task = npz['task'].item()
n_iterations = check_freq * length
batch_size = npz['settings_batch_size']

description = 'task: {}\ntraining time: {:2.2f} min,  num iterations: {:n}, batch_size: {}\nactivation fnc: {}  ' \
              'n_hidden: {:d}  n_in: {:d}  n_out: {:d}\n'.format(task, elapsed_time / 60, n_iterations, batch_size,
                                                                 activation_fnc, n_hidden, n_in, n_out)
axarr[0].set_title(description, fontsize=14, ha='left', multialignment='left', loc='left')

plt.xlabel('iteration num')
plt.grid(True)
plt.savefig("plot.svg")
plt.show()
