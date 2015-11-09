import numpy
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as TT
import theano as T

srng = RandomStreams(seed=13)
u = srng.uniform(low=0, high=1, size=(200, 1))

exp_lambda = 1
x = -TT.log(-u+1)/exp_lambda
r = x/x.sum()

f = T.function([], r)


for i in range(300000):
    a = f()
    if sum(a) < 0.9:
        print('MERDA')
