
import numpy
__author__ = 'giulio'

n_hidden = 100
rng = numpy.random.RandomState(13)
W_rec = numpy.asarray(rng.normal(size=(n_hidden, n_hidden), scale=.1, loc = .0), dtype = 'float32')


rho = numpy.max(abs(numpy.linalg.eigvals(W_rec)))
print('RHO: {}'.format(rho))