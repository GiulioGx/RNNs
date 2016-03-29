import numpy
import matplotlib.pyplot as plt
import sys

from matplotlib.ticker import FormatStrFormatter

from plotUtils.plt_utils import save_multiple_formats

__author__ = 'giulio'

"""This script plots train and validation losses for models trained with different number of hidden units"""


lengths = [10, 20, 50, 100, 150, 200]
rates_rho = [100, 100, 100, 66, 66, 0]
rates_old = [100, 100, 0, 0, 0, 0]

plt.plot(lengths, rates_rho, '--o', color='b', linewidth=1)
plt.plot(lengths, rates_old, '--o', color='r', linewidth=1)

plt.legend(["SGD-C rho>1","SGD-C rho<1"], shadow=True, fancybox=True)



# plt.yscale('log')
# plt.ylim(ymin=4, ymax=13)
plt.xticks(lengths)
plt.xlim(xmin = 0, xmax = 220)
plt.yticks(numpy.arange(11)*10)
plt.ylim(ymin=-10, ymax = 110)
# plt.legend(legends, shadow=True, fancybox=True)
plt.xlabel('lengths')
plt.ylabel('rate of success')

# ax = plt.gca()
# ax.xaxis.set_major_formatter(FormatStrFormatter("%.0e"))
# formatter = FormatStrFormatter("%.1f")
# # formatter.set_useOffset(True)
# ax.yaxis.set_major_formatter(formatter)
# # ax.xaxis.get_major_formatter().set_useOffset(False)
filename = sys.argv[0]
save_multiple_formats(filename)

ax = plt.gca()
ax.set_xmargin(1)
plt.show()
