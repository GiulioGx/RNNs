from ActivationFunction import ActivationFunction, Tanh
from infos.InfoProducer import SimpleInfoProducer
from output_fncs import OutputFunction
from output_fncs.Linear import Linear
from model.RNNInitializer import RNNInitializer
from model.RNN import RNN


class RNNBuilder(SimpleInfoProducer):
    def __init__(self, initializer: RNNInitializer, n_hidden: int = 100,
                 activation_fnc: ActivationFunction = Tanh(), output_fnc: OutputFunction = Linear):
        self.__initializer = initializer
        self.__activation_fnc = activation_fnc
        self.__output_fnc = output_fnc
        self.__n_hidden = n_hidden

    def init_net(self, n_in: int, n_out: int):
        W_rec, W_in, W_out, b_rec, b_out = self.__initializer.generate_variables(n_in=n_in, n_out=n_out, n_hidden=self.__n_hidden)
        rnn = RNN(W_rec=W_rec, W_in=W_in, W_out=W_out, b_rec=b_rec, b_out=b_out, activation_fnc=self.__activation_fnc,
                  output_fnc=self.__output_fnc, variables_initializer=self.__initializer)
        return rnn

    @property
    def activation_fnc(self):
        return self.__activation_fnc

    @property
    def infos(self):
        return self.__initializer.infos
