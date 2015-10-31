import abc

__author__ = 'giulio'


class Params(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __add__(self, other):
        """addition with params operation"""

    @abc.abstractmethod
    def __sub__(self, other):
        """subtraction with params operation"""

    @abc.abstractmethod
    def __mul__(self, alpha):
        """scalar multiplication"""

    @abc.abstractmethod
    def dot(self, other):
        """dot product between two param objects viewed as vectors"""

    @abc.abstractmethod
    def norm(self):
        """euclidean norm"""

    @abc.abstractmethod
    def grad(self, fnc):
        """return the gradient of the function fnc wrt the parameters this class represents"""

    @abc.abstractmethod
    def grad_combining_steps(self, loss_fnc, strategy, u, t):
        """return a combinantion of gradients for each time steps defined by 'strategy' of the function 'fnc'
        wrt the parameters this class represents"""

    @abc.abstractmethod
    def update_dictionary(self, other):
        """return the update dictionary for a theano function"""
