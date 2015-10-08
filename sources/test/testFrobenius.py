import theano as T
import theano.tensor as TT
import numpy


__author__ = 'giulio'



W = TT.matrix()
deriv_a = TT.fcol()
A = W * deriv_a
frobenius_norm = (A**2).sum()
reg = TT.grad(frobenius_norm, [W], consider_constant=[deriv_a])


f = T.function([W,deriv_a], reg)



w = numpy.matrix([[1, 2],[3, 4]], dtype='float32')
d = numpy.matrix([[3], [5]], dtype='float32')

print(w)
print(w.shape)
print(d)

print(f(w,d))