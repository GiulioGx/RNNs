import theano as T
import theano.tensor as TT
import numpy
from theanoUtils import is_not_trustworthy

H = TT.matrix()
norm_G = H.norm(2, axis=1).reshape((H.shape[0], 1))
G = H / TT.switch(is_not_trustworthy(norm_G), 1, norm_G)

f = T.function([H], [G, norm_G],allow_input_downcast=True)

h = numpy.array([[1, 1.], [3, 4]], dtype='float32')
g, norm_g = f(h)

print('g:', g,'\nnorm_g:', norm_g)
