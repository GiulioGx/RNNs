
import numpy
import theano.tensor as TT
import  theano as T

from theanoUtils import as_vector

a = TT.matrix()
b = TT.matrix()



v = TT.addbroadcast(as_vector(a, b), 1)

d = v[0:6]
d1 = d.reshape((3, 2))

f = T.function([a, b], [v, d, d1])


v1 = numpy.asarray([[1, 10], [3, 30], [7, 70]], dtype='float32')
v2 = numpy.asarray([[2, 20], [5, 50], [9, 90]], dtype='float32')


print(v1)
print('###')
v, v_r, v_1 = f(v1, v2)
print(v)
print(v_r)
print(v_1)


