from configs import Configs
import theano as T
import theano.tensor as TT
import numpy

__author__ = 'giulio'



b_out = numpy.zeros( (3, 1), Configs.floatType)
b2 = numpy.ones( (3,1), Configs.floatType)

x = T.shared(b_out, 'b_out', broadcastable=(False, True))


def step(beta, d1, x):
    return beta/2, {x: d1}

d1 = TT.fcol()


beta = TT.scalar(dtype=Configs.floatType)
values, updates = T.scan(step, outputs_info=beta, non_sequences=[d1, x], n_steps=3, strict='true')

f = T.function([beta, d1], [values], updates=updates)
print(f(3.0, b2))
print(x.get_value())
print(x.shape)