import numpy

run_cheked_val = [1075800, 1969200, 1847000]
run_cos_vals = [1659000, 3018000]
run_anti = [1538800, 1853800, 2029800]

print("mean_cos: {}".format(int(numpy.mean(run_cos_vals))))
print("mean_checked: {}".format(int(numpy.mean(run_cheked_val))))
print("mean_anti: {}".format(int(numpy.mean(run_anti))))


# antigradient
temp_50 = [211200, 827600, 58200, 175400, 738200]
temp_100 = [2396800, 1932800, 'x']
temp_150 = [3529800, ]

temp_100_simplex = [973000, 923400, 1133600]

print('tOrderTask...')
print('mean_anti: {}'.format(int(numpy.mean(temp_100[0:2]))))
print("mean_checked: {}".format(int(numpy.mean(temp_100_simplex))))
