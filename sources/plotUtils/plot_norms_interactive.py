import numpy
import matplotlib.pyplot as plt

__author__ = 'giulio'

modelFile = '/home/giulio/RNNs/models/model_add.npz'
#modelFile = '/home/giulio/model_octopus.npz'
npz = numpy.load(modelFile)
norms_dicts = npz['obj_separate_norms']
check_freq = npz['settings_check_freq']
length = npz['length']

sep = '#'*5


def on_button_press(event):
    plt.close("all")


x = range(length)
# reversed(x)

for i in reversed(x):
    dict = norms_dicts[i]
    keys = sorted(dict.keys())
    fig, axarr = plt.subplots(len(dict), sharex=True, figsize=(20, 30))

    j = 0
    for key in keys:
        y = dict[key]
        axarr[j].bar(range(len(y)), y)
        axarr[j].legend([key], shadow=True, fancybox=True)
        axarr[j].set_yscale('log')
        j += 1

    cid = fig.canvas.mpl_connect('key_press_event', on_button_press)
    print(sep)
    print('Press a button to continue...')
    fig.canvas.set_window_title('Gradient norms it: {:07d}'.format(i*check_freq))
    plt.show()




