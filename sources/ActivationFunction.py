from Configs import Configs
import theano.tensor as TT
import numpy
import abc

__author__ = 'giulio'


class ActivationFunction(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def f(self, x):
        """applies f() to a theano symbol x"""

    def grad_f(self, x):
        """applies f'() to a theano symbol x"""

    def __str__(self):
        return 'unkwown'


# predefined activation functions
class Relu(ActivationFunction):

    def f(self, x):
        return TT.switch(x < 0, 0, x)

    def grad_f(self, x):
        return TT.switch(x > 0, TT.alloc(numpy.array(1., dtype=Configs.floatType)),
                         TT.alloc(numpy.array(0., dtype=Configs.floatType)))

    def __str__(self):
        return 'relu'


class Tanh(ActivationFunction):

    def f(self, x):
        return TT.tanh(x)

    def grad_f(self, x):
        return 1 - (TT.tanh(x) ** 2)

    def __str__(self):
        return 'tanh'


class Identity(ActivationFunction):
    def f(self, x):
        return x

    def grad_f(self, x):
        return TT.ones_like(x) # XXX

    def __str__(self):
        return 'identity'

# TODO sigmoid
