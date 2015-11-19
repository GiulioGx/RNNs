import numpy
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as TT
import theano as T

srng = RandomStreams(seed=13)
u = srng.choice(size=10, a=[0, 1], replace=True, p=[0.2, 0.8], ndim=1, dtype='int32')


f = T.function([],u)

print(f())


