import theano as T
import theano.tensor as TT
import numpy
from configs import Configs

__author__ = 'giulio'


x = TT.tensor3()

b = TT.switch(x < 0, 0, x)

d = TT.grad()

deriv = TT.switch(x > 0, TT.alloc(numpy.array(1., dtype=Configs.floatType)),
                                TT.alloc(numpy.array(0., dtype=Configs.floatType)))

f = T.function([x], [b, deriv])

w = [[1, -1], [0, 5]]

y, v = f(w)
print(y)
print(v)