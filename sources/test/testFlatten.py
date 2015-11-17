import numpy
import theano.tensor as TT
import theano as T
from theano.tensor import nlinalg as li
from theano.tensor import slinalg as sli
from theano.tensor.shared_randomstreams import RandomStreams

from combiningRule.SimplexCombination import SimplexCombination
from model import RnnGradient
from theanoUtils import as_vector, flatten_list_element

a1 = TT.matrix()
a2 = TT.matrix()
b1 = TT.dcol()
b2 = TT.dcol()

a = TT.as_tensor_variable([a1, a2])
b = TT.as_tensor_variable([b1, b2])

b = RnnGradient.fix(b, 2)

c = flatten_list_element([a, b], 2)

rule = SimplexCombination()
symbols = rule.compile(c, 2)

v = symbols.combination
alphas = symbols.infos

#from flattend
n_hidden = 2
n1 = n_hidden ** 2
n2 = n1 + n_hidden

W_rec_v = v[0:n1]
b_rec_v = v[n1:n2]

W_rec = W_rec_v.reshape((n_hidden, n_hidden))
b_rec = b_rec_v.reshape((n_hidden, 1))

f = T.function([a1, a2, b1, b2], [c, TT.as_tensor_variable(v), TT.as_tensor_variable(alphas), W_rec, b_rec])

a1_ = [[1, 2], [3, 4]]
b1_ = [[5], [6]]
a2_ = [[9, 10], [11, 12]]
b2_ = [[13], [14]]


c_, v_, alphas_, W_rec, b_rec = f(a1_, a2_, b1_, b2_)
alphas_ = alphas_[0]
print('shape', alphas_.shape)
print('c[0]:', c_[0])
print('c[1]:', c_[1])

print('v:', v_)
print('alphas', alphas_)

print('norm_0: ', numpy.linalg.norm(c_[0], 2))
print('norm_v: ', numpy.linalg.norm(v_, 2))

print('gt:', c_[0]/numpy.linalg.norm(c_[0], 2)*alphas_[0] + c_[1]/numpy.linalg.norm(c_[1], 2)*alphas_[1])

print('W_rec', W_rec)
print('b_rec', b_rec)

print('testprec')


srng = RandomStreams(seed=13)
u = srng.uniform(low=0, high=1, size=(200, 1))

# x = TT.log(1.-u)
# r = x/x.sum()

d = srng.normal(size=(200, 1))

r = d/d.norm(2)

x = TT.exp(1.-u)
r = x/x.sum()

comb = T.function([], r)

betas = numpy.random.rand(200)



