from theanoUtils import normalize
from theanoUtils import norm
import theano as T
import numpy

__author__ = 'giulio'



import theano.tensor as TT

v = TT.vector()


r = normalize(v)
norm1 = norm(r)
f = T.function([v], norm1)

alpha = (1/ norm(v))
r2 = v * alpha
norm2 = norm(r2)
g = T.function([v], [norm2, alpha, norm(v)])


input = numpy.array([0.0000000000000000000000000000000003, 0.00000000000000000000000000000000000002], dtype='float32')
print(f(input))
print(g(input))





