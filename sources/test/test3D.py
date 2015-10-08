from configs import Configs
import theano as T
import theano.tensor as TT
import numpy
__author__ = 'giulio'





W = TT.tensor3()
A = TT.tensor3()



res = TT.dot(W,A)

f = T.function([W, A], res)


w = numpy.zeros((3, 3, 2), dtype=Configs.floatType)

w[:, :, 0] = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
w[:, :, 1] = [[5, 5, 5], [6, 6, 6], [7, 7, 7]]

print(w.shape)


a = numpy.zeros((3, 2, 3), dtype=Configs.floatType)
a[:, 0, :] = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
a[:, 1, :] = [[5, 5, 5], [6, 6, 6], [7, 7, 7]]

print(a.shape)


c = f(w, a)

print(c.shape)
