import numpy
import matplotlib.pyplot as plt
import sys

from plotUtils.plt_utils import save_multiple_formats

run13 = [40000, 80000, 1098600, 1250000]
runs = [run13]

mat = numpy.array(runs)
means = numpy.mean(mat, axis=0)
x = [50, 100, 150, 200]

width = 20
plt.bar(x, means, width=width, align='center')
plt.yscale('log')
plt.xticks(x)
# plt.legend(legends, shadow=True, fancybox=True)
plt.xlabel('length')
plt.ylabel('iterations')
filename = sys.argv[0]
save_multiple_formats(filename)
plt.show()
