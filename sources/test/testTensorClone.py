import numpy
import theano as T
import theano.tensor as TT

u = TT.tensor3('u')
W = T.shared(name='W', value=numpy.ones(shape=(20, 20)))


def fill(W):
    return W

values, updates_scan = T.scan(fill, non_sequences=[W],
                              outputs_info=[None],
                              name='replicate_scan', n_steps=u.shape[0])

W3 = values


def step(u_t, W):
    a_t = TT.dot(W, u_t)

    return a_t

values, updates_scan = T.scan(step, sequences=[u, W3],
                              outputs_info=[None],
                              name='a_scan')

result = values[0].sum().sum()
grad = T.grad(result, wrt=W3)

# then i would like to call theano.grad on W_copy (I think I can do it :))

f = T.function([u], [W3],
               on_unused_input='warn', updates=updates_scan)

u = numpy.ones(shape=(10, 20, 5))
fu = f(u)[0]
print(fu.shape)
print(fu[1])
