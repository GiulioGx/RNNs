import numpy

run_cheked_val = [1075800, 1969200, 1847000]
run_cos_vals = [1659000, 3018000]
run_anti = [1538800, 1853800, 2029800]

print("mean_cos: {}".format(int(numpy.mean(run_cos_vals))))
print("mean_checked: {}".format(int(numpy.mean(run_cheked_val))))
print("mean_anti: {}".format(int(numpy.mean(run_anti))))
