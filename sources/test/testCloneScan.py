import numpy
import theano as T
import theano.tensor as TT

u = TT.tensor3('u')
W = T.shared(name='W', value=numpy.ones(shape=(20, 20)))


def step(u_t, W):
    W_copy = W.clone() # without the clone works
    a_t = TT.dot(W_copy, u_t)

    return W_copy, a_t

values, updates_scan = T.scan(step, sequences=[u],
                              outputs_info=[W, None],
                              name='a_scan')

result = values[0].sum().sum()

# then i would like to call theano.grad on W_copy (I think I can do it :))

f = T.function([u], [result],
               on_unused_input='warn', updates=updates_scan)

u = numpy.ones(shape=(10, 20, 5))
print(f(u))
