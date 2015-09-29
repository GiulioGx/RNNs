import theano as T
import theano.tensor as TT
import numpy

__author__ = 'giulio'

u = TT.tensor3()
n_sequences = u.shape[2]
h_0 = TT.alloc(numpy.array(0., dtype='float32'), 4, n_sequences)


def h_t(u_t, h_tm1):
    return u_t + h_tm1


h, _ = T.scan(h_t, sequences=u,
              outputs_info=[h_0],
              non_sequences=[],
              name='h_t',
              mode=T.Mode(linker='cvm'))

f = T.function([u], h)

c1 = numpy.zeros((3, 4, 2), dtype='float32')
c1[:, :, 0] = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
c1[:, :, 1] = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]

y = f(c1)

print(y.shape)
print(c1[:, :, 0])
print(y[:, :, 0])
print(c1[:, :, 1])
print(y[:, :, 1])


b = TT.tensor3()


def y_t(b_t):
    return b_t + 3

y, _ = T.scan(y_t, sequences=b,
              outputs_info=[None],
              non_sequences=[],
              name='y_t',
              mode=T.Mode(linker='cvm'))

g = T.function([b], y)

c2 = numpy.zeros((3, 1, 2), dtype='float32')
c2[:, :, 0] = [[1], [10], [20]]
c2[:, :, 1] = [[1], [2], [3]]

m = g(c2)

print(m.shape)
print(c2[:, :, 0])
print(m[:, :, 0])
print(c2[:, :, 1])
print(m[:, :, 1])


