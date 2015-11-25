import abc

__author__ = 'giulio'


class Variables(object):
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
    def __neg__(self):
        """negation: returns -self"""

    @abc.abstractmethod
    def dot(self, other):
        """dot product between two param objects viewed as vectors"""

    @abc.abstractmethod
    def norm(self):
        """euclidean norm"""

    @abc.abstractmethod
    def cos(self, other):
        """returns the cosine between this 'Variable' object (seen as a vector) and 'other' """

    @abc.abstractmethod
    def failsafe_grad(self, loss_fnc, u, t):
        """return the gradient of the function fnc wrt the parameters this class represents"""

    @abc.abstractmethod
    def gradient(self, loss_fnc, u, t):
        """return a class that can produce combinantion of gradients for each time steps given a 'strategy' for combining them
        wrt the parameters this class represents"""

    @abc.abstractmethod
    def net_output(self, u):
        """returns the output of the network parametrized with the variables this class represents """

    @abc.abstractmethod
    def update_list(self, other):
        """returns the update dictionary for a theano function"""

    @abc.abstractmethod
    def as_tensor(self):
        """return a tensor view of this object"""

    @abc.abstractproperty
    def net(self):
        """returns the networks this 'Variables' class parametrizes"""
