import matplotlib.pyplot as plt
import numpy

__author__ = 'giulio'

stats_file = '/home/giulio/tmp_models/stats.npz'

# load npz archive
npz = numpy.load(stats_file)

# choose what and how to display it
valid_error = {'label': 'validation_error_curr', 'legend': 'validation error', 'color': 'r', 'scale': 'linear'}
valid_loss = {'label': 'validation_loss', 'legend': 'validation loss', 'color': 'm', 'scale': 'log'}
train_loss = {'label': 'train_loss', 'legend': 'train loss', 'color': 'b', 'scale': 'log'}
dir_norm = {'label': 'dir_norm', 'legend': 'dir norm', 'color': 'c', 'scale': 'log'}
grad_dot = {'label': 'grad_dot', 'legend': 'gradient dot', 'color': 'b', 'scale': 'linear'}
equi_cos = {'label': 'W_rec_equi_cos', 'legend': 'equi_cos', 'color': 'b', 'scale': 'linear'}
rho = {'label': 'rho', 'legend': 'W_rec spetral radius', 'color': 'r', 'scale': 'linear'}
grad_var = {'label': 'obj_g_var', 'legend': 'gradient var', 'color': 'g', 'scale': 'linear'}
dots_var = {'label': 'obj_dots_var', 'legend': 'dots var', 'color': 'b', 'scale': 'linear'}


# only elements in this list gets displayed, elements in the same list get displayed togheter
measures = [(train_loss, valid_loss)]

# compute x-axis points
check_freq = npz['settings_check_freq']
length = npz['length']
x_values = numpy.arange(length) * check_freq

# plot
n_plots = len(measures)
fig, axarr = plt.subplots(n_plots, sharex=True)

for i in range(n_plots):
    measure = measures[i]
    ax = axarr[i] if n_plots > 1 else axarr
    legends = []
    for m in measure:
        print(m)
        y_values = npz[m['label']]
        ax.plot(x_values, y_values, m['color'])
        ax.set_yscale(m['scale'])
        legends.append(m['legend'])

    ax.legend(legends, shadow=True, fancybox=True)



# description
elapsed_time = npz['elapsed_time'].item()
n_hidden = npz['net_n_hidden'].item()
n_in = npz['net_n_in'].item()
n_out = npz['net_n_out'].item()
activation_fnc = npz['net_activation_fnc']
task = npz['datasets'].item()
n_iterations = check_freq * length
batch_size = npz['settings_batch_size']

description = 'datasets: {}\ntraining time: {:2.2f} min,  num iterations: {:n}, batch_size: {}\nactivation fnc: {}  ' \
              'n_hidden: {:d}  n_in: {:d}  n_out: {:d}\n'.format(task, elapsed_time / 60, n_iterations, batch_size,
                                                                 activation_fnc, n_hidden, n_in, n_out)
ax = axarr[0] if n_plots > 1 else axarr
ax.set_title(description, fontsize=14, ha='left', multialignment='left', loc='left')

plt.xlabel('iteration num')
plt.grid(True)
plt.savefig("plot.svg")
plt.show()
