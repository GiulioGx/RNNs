import matplotlib.pyplot as plt
import numpy


x = numpy.arange(-10, 10, 0.001)
y_values = 2 * numpy.cos(x) + numpy.cos(2*x-0.5)
y2 = -numpy.sin(x) - numpy.sin(2*x-0.5)
# plot
n_plots = 2
fig, axarr = plt.subplots(n_plots, sharex=True)
axarr[0].plot(x, y_values, 'r')
axarr[1].plot(x, y2, 'b')
plt.grid(True)
plt.show()