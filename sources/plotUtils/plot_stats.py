import matplotlib.pyplot as plt
import numpy

__author__ = 'giulio'

modelFile = '/home/giulio/RNNs/models/model_add.npz'
#modelFile = '/home/giulio/model_octopus.npz'


npz = numpy.load(modelFile)
valid_error = npz['validation_error_curr']
#penalty = npz['penalty']
grad_norm = npz['obj_loss_grad']
grad_dot = npz['grad_dot']


check_freq = npz['settings_check_freq']
length = len(valid_error) #FIXME
x_values = numpy.arange(length) * check_freq

fig, axarr = plt.subplots(3, sharex=True)

axarr[0].plot(x_values, valid_error, 'r')
axarr[0].legend(['validation error'], shadow=True, fancybox=True)

# axarr[1].plot(x_values, penalty, 'c')
# axarr[1].legend(['penalty grad norm'], shadow=True, fancybox=True)
# axarr[1].set_yscale('log')

axarr[1].plot(x_values, grad_norm, 'm')
axarr[1].legend(['gradient norm'], shadow=True, fancybox=True)
axarr[1].set_yscale('log')

axarr[2].plot(x_values, grad_dot, 'b')
axarr[2].legend(['gradient dot'], shadow=True, fancybox=True)

# description
elapsed_time = npz['elapsed_time'].item()
n_hidden = npz['n_hidden'].item()
n_in = npz['n_in'].item()
n_out = npz['n_out'].item()
activation_fnc = npz['activation_fnc']
# task = npz   # FIXME
task = 'todo'
n_iterations = check_freq * length

description = 'task: {}\ntraining time: {:2.2f} min  num iterations: {:n}\nactivation fnc: {}  ' \
              'n_hidden: {:d}  n_in: {:d}  n_out: {:d}\n'.format(task, elapsed_time / 60, n_iterations,
                                                                 activation_fnc, n_hidden, n_in, n_out)
axarr[0].set_title(description, fontsize=14, ha='left', multialignment='left', loc='left')

plt.xlabel('iteration num')
plt.grid(True)
plt.savefig("test.svg")
plt.show()
