import numpy
import matplotlib.pyplot as plt

__author__ = 'giulio'

modelFile = '/home/giulio/RNNs/models/temporal_order_plain, min_length: 50_14/stats.npz'
npz = numpy.load(modelFile)
norms_dicts = npz['obj_separate_norms']
check_freq = npz['settings_check_freq']
length = npz['length']
temporal_dots = npz['obj_grad_temporal_cos']

sep = '#' * 5


def on_button_press(event):
    plt.close("all")


x = range(length)
#x = reversed(x)

for i in x:
    dict = norms_dicts[i]
    keys = sorted(dict.keys())
    print(keys)
    fig, axarr = plt.subplots(len(dict) + 1, sharex=True, figsize=(20, 30))

    j = 0
    for key in keys:
        print(key)
        y = dict[key]
        y = y if y.ndim == 1 else y[0]
        print(y)
        axarr[j].bar(range(len(y)), y)
        axarr[j].legend([key], shadow=True, fancybox=True)
        if any(y > 0):
            axarr[j].set_yscale('log')
        j += 1

    y = temporal_dots[i]
    axarr[j].bar(range(len(y)), y, color='r')
    axarr[j].legend(['temporal_cos'], shadow=True, fancybox=True)

    cid = fig.canvas.mpl_connect('key_press_event', on_button_press)
    print(sep)
    print('Press a button to continue...')
    fig.canvas.set_window_title('Gradient norms it: {:07d}'.format(i * check_freq))

    # fig, axarr = plt.subplots(1, sharex=True, figsize=(20, 30))
    # y = variance[i]
    # axarr.bar(range(len(y)), y, color='m')
    # axarr.legend(['variance'], shadow=True, fancybox=True)
    # fig.canvas.set_window_title('grad_variance: {:07d}'.format(i*check_freq))

    plt.show()
