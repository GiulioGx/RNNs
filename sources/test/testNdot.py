import theano as T
import theano.tensor as TT
import numpy


__author__ = 'giulio'



W = TT.matrix()

a = TT.matrix()

reg = W * a

f = T.function([W, a], reg)



w = numpy.matrix([[1, 2],[3, 4]], dtype='float32')
d = numpy.matrix([[3], [5]], dtype='float32')

print(w)
print(w.shape)
print(d)

print(f(w,d))