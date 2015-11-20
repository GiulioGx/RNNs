import numpy
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as TT
import theano as T

srng = RandomStreams(seed=13)
u = srng.uniform(low=0, high=1, size=(5, 1))

x = TT.exp(1.-u)
r = x/x.sum()

u/u.sum()

f = T.function([], [r, u])


for i in range(5):
    a, u = f()
    print(a)
    print('u', u)
    #print('sum: {}'.format(sum(a)))
    if sum(a) < 0.9:
        print('MERDA')
