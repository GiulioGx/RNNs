import abc
import theano as T
import theano.tensor as TT
from ActivationFunction import ActivationFunction
from Configs import Configs
from infos.Info import NullInfo
from initialization.MatrixInit import MatrixInit

__author__ = 'giulio'


class NeuralNetwork(object):
    __metaclass__ = abc.ABCMeta
    """defines the abstract type 'Neural Network'"""

    @abc.abstractproperty
    def n_in(self):
        """"""

    @abc.abstractproperty
    def n_out(self):
        """"""

    @abc.abstractproperty
    def n_variables(self):
        """"""

    @abc.abstractmethod
    def net_ouput_numpy(self, u):  # TODO names Theano T
        """"""

    @abc.abstractmethod  # schould be a template method
    def net_output(self, params: RNNVars, u, h_m1):
        """"""

    def from_tensor(self, v):
        """"""

    @abc.abstractproperty
    def info(self):
        """"""

    @abc.abstractmethod
    def save_model(self, filename: str):
        """saves the model with statistics to file"""

    @staticmethod
    def load_model(filename):  # TODO
        """"""

    @abc.abstractproperty
    def spectral_info(self):
        """"""


class CustomNeuralNetwork(NeuralNetwork):
    def __init__(self, n_in: int, n_out: int):
        self.__layers = []

    def add_layer(self, layer: Layer):
        pass

    def set_output_fnc(self, fnc):
        pass


class AbstractLayer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def variables(self) -> list:
        """return the list of the shared variables used in the computation"""

    @abc.abstractproperty
    def n_in(self) -> int:
        """return the dimensionality of the input"""

    @abc.abstractmethod
    def n_out(self) -> int:
        """return the dimensionality of the output"""

    @abc.abstractmethod
    def output(self, recurrent_output, params):


class Layer(AbstractLayer):
    def __init__(self, size, weights_init_strategy: MatrixInit, bias_init_strategy: MatrixInit,
                 activation_fnc: ActivationFunction):
        # define shared variables
        self.__id = 0  # TODO
        self.__W = T.shared(weights_init_strategy.init_matrix(size=size, dtype=Configs.floatType),
                            name='W_' + str(self.__id))
        self.__activation_fnc = activation_fnc

    @property
    def variables(self) -> list:
        pass

    def n_in(self) -> int:
        pass

    def n_out(self) -> int:
        pass
