from configs import Configs
import theano as T
import theano.tensor as TT
import numpy
__author__ = 'giulio'


a = TT.tensor3()
c = a.sum(axis=0)
f = T.function([a], c)


w = numpy.matrix([[[1, 2],[3, 4]],[[1, 2],[3, 4]] ], dtype='float32')
d = numpy.matrix([[3, 5]], dtype='float32')

print(f(w))