import numpy
from plotUtils.plot_fncs import plot_norms
import matplotlib.pyplot as plt

__author__ = 'giulio'

modelFile = '/home/giulio/RNNs/models/model.npz'

npz = numpy.load(modelFile)
norms = npz['separate_norms']
check_freq = npz['settings_check_freq']
length = npz['length']

print(check_freq)
print(type(check_freq))

sep = '#'*5


def on_button_press(event):
    plt.close("all")

for i in range(length):
    fig, ax = plot_norms(norms[i])
    cid = fig.canvas.mpl_connect('key_press_event', on_button_press)
    print(sep)
    print('Press a button to continue...')
    fig.canvas.set_window_title('Gradient norms it: {:07d}'.format(i*check_freq))
    ax.set_yscale('log')
    plt.show()




