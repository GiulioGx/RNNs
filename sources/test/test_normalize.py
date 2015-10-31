from theanoUtils import normalize
from theanoUtils import norm
import theano as T
import numpy

__author__ = 'giulio'

import theano.tensor as TT
import theano

# bnum = numpy.array([[1], [3]], dtype='float32')
# b = theano.shared(bnum, broadcastable=(False, True))

bnum = numpy.array([[1,2], [3,3]], dtype='float32')
b = theano.shared(bnum)


b_fixes = []
for i in range(200):  # FIXME max_lenght
    b_fixes.append(b.clone())


def step(v, acc):
        return v + acc, norm(v)

values, _ = T.scan(step, sequences=[TT.as_tensor_variable(b_fixes)],
                           outputs_info=[TT.unbroadcast(TT.zeros_like(b_fixes[0]), 1), None],
                           non_sequences=[],
                           name='net_output',
                           mode=T.Mode(linker='cvm'))

grads_combinantions = values[0]
separate_norms = values[1]

r = grads_combinantions[-1]
TT.switch(b.shape[])
#TT.addbroadcast(r,1)

f = theano.function([], [grads_combinantions[-1], grads_combinantions[-1].shape, b.shape], updates=[(b, r)])
g = theano.function([], TT.as_tensor_variable(b_fixes))
a,s1,s2 = f()
print('s1: '+str(s1)+'s2: '+str(s2))
print(a)
c = g()
print(c.shape)
print(bnum.shape)