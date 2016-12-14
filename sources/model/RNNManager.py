from ActivationFunction import ActivationFunction, Tanh
from model.NetManager import NetManager
from model.RNNGrowingPolicy import RNNGrowingPolicy, RNNNullGrowing
from output_fncs import OutputFunction
from output_fncs.Linear import Linear
from model.RNNInitializer import RNNProducer
from model.RNN import RNN


class RNNManager(NetManager):
    def __init__(self, initializer: RNNProducer,
                 activation_fnc: ActivationFunction = Tanh(), output_fnc: OutputFunction = Linear,
                 growing_policy: RNNGrowingPolicy = RNNNullGrowing()):
        self.__initializer = initializer
        self.__activation_fnc = activation_fnc
        self.__output_fnc = output_fnc
        self.__growing_policy = growing_policy
        self.__net = None

    def get_net(self, n_in: int, n_out: int):
        W_rec, W_in, W_out, b_rec, b_out = self.__initializer.generate_variables(n_in=n_in, n_out=n_out)

        rnn = RNN(W_rec=W_rec, W_in=W_in, W_out=W_out, b_rec=b_rec, b_out=b_out, activation_fnc=self.__activation_fnc,
                  output_fnc=self.__output_fnc)

        self.__net = rnn
        return rnn

    def grow_net(self, logger):
        self.__growing_policy.grow(self.__net, logger)

    @property
    def activation_fnc(self):
        return self.__activation_fnc

    @property
    def infos(self):
        return self.__initializer.infos
