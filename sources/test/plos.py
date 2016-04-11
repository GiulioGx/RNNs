import numpy
import matplotlib.pyplot as plt


plt.figure()

x = numpy.arange(-1, 0, 0.01)
y = 1. / x

plt.plot(x,y)
plt.show()