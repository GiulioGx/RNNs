import numpy
from plotUtils.plot_fncs import plot_norms
import matplotlib.pyplot as plt

__author__ = 'giulio'

modelFile = '/home/giulio/RNNs/models/model.npz'

npz = numpy.load(modelFile)
norms = npz['separate_norms']
length = npz['length']


def on_button_press(event):
    plt.close("all")

for i in range(length):
    fig = plot_norms(norms[i])
    cid = fig.canvas.mpl_connect('key_press_event', on_button_press)
    print('Press a button to continue...')
    fig.canvas.set_window_title('Gradient norms it: {:07d}'.format(i*50))  # FIXME
    plt.show()




