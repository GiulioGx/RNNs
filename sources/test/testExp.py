import numpy

rng = numpy.random.RandomState(12)
u = rng.uniform(low=0, high=1, size=(1, 5))

lamda = 1
x = -numpy.log(-u+1)/lamda

print(u)
print(x)
print(numpy.sum(x))
r = x/numpy.sum(x)
print(r)
print(numpy.sum(r))
