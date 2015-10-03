from configs import Configs
import theano as T
import theano.tensor as TT
import numpy
__author__ = 'giulio'



a = TT.matrix()
b = TT.matrix()


#c = TT.stacklists([a.flatten(1), b.flatten(1)]).flatten()
c1 = a.flatten()
c2 = b.flatten()
c3 = TT.concatenate([c1,c2])

f = T.function([a, b], [c3])

at = [[1,2, 3], [2,3,3]]
bt = [[8,7], [6,5]]

print(f(at,bt))