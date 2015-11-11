import numpy
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as TT
import theano as T
from theano.compile.nanguardmode import NanGuardMode

srng = RandomStreams(seed=13)
u = srng.uniform(low=0, high=1, size=(250, 1))

# x = TT.log(1.-u)
# r = x/x.sum()

d = srng.normal(size=(200, 1))

r = d/d.norm(2)

x = TT.exp(1.-u)
r = x/x.sum()

f = T.function([], r, mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False))


for i in range(300000):
    a = f()
    #print('sum: {}'.format(sum(a)))
    if sum(a) < 0.9:
        print('MERDA')
