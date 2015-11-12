
import numpy
import theano.tensor as TT
import  theano as T
from theano.tensor import nlinalg as li
from theano.tensor import slinalg as sli


from theanoUtils import as_vector

a = TT.matrix()


e = li.eig(a)
rho = TT.max(abs(e[0]))

f = T.function([a], rho)



m = [[1, 0], [0, -7]]
print(f(m))


