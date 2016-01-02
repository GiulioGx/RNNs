from ActivationFunction import ActivationFunction, Tanh
from Configs import Configs
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from infos.SimpleInfoProducer import SimpleInfoProducer
from initialization.GaussianInit import GaussianInit
from initialization.MatrixInit import MatrixInit
from model.Rnn import Rnn
from output_fncs import OutputFunction
from output_fncs.Linear import Linear


class RnnInitializer(SimpleInfoProducer):
    def __init__(self, W_rec_init: MatrixInit = GaussianInit(), W_in_init: MatrixInit = GaussianInit(),
                 W_out_init: MatrixInit = GaussianInit(), b_rec_init: MatrixInit = GaussianInit(),
                 b_out_init: MatrixInit = GaussianInit(), n_hidden: int = 100,
                 activation_fnc: ActivationFunction = Tanh(), output_fnc: OutputFunction = Linear):
        self.__W_rec_init = W_rec_init
        self.__W_in_init = W_in_init
        self.__W_out_init = W_out_init
        self.__b_rec_init = b_rec_init
        self.__b_out_init = b_out_init
        self.__n_hidden = n_hidden
        self.__activation_fnc = activation_fnc
        self.__output_fnc = output_fnc

    def init_net(self, n_in: int, n_out: int):
        # init network matrices
        W_rec = self.__W_rec_init.init_matrix((self.__n_hidden, self.__n_hidden), Configs.floatType)
        W_in = self.__W_in_init.init_matrix((self.__n_hidden, n_in), Configs.floatType)
        W_out = self.__W_out_init.init_matrix((n_out, self.__n_hidden), Configs.floatType)

        # init biases
        b_rec = self.__b_rec_init.init_matrix((self.__n_hidden, 1), Configs.floatType)
        b_out = self.__b_out_init.init_matrix((n_out, 1), Configs.floatType)
        rnn = Rnn(W_rec=W_rec, W_in=W_in, W_out=W_out, b_rec=b_rec, b_out=b_out, activation_fnc=self.__activation_fnc,
                  output_fnc=self.__output_fnc)
        return rnn

    @property
    def activation_fnc(self):
        return self.__activation_fnc

    @property
    def infos(self):
        W_rec_info = InfoGroup('W_rec', InfoList(self.__W_rec_init.infos))
        W_in_info = InfoGroup('W_in', InfoList(self.__W_in_init.infos))
        W_out_info = InfoGroup('W_out', InfoList(self.__W_out_init.infos))
        b_rec_info = InfoGroup('b_rec', InfoList(self.__b_rec_init.infos))
        b_out_info = InfoGroup('b_out', InfoList(self.__b_out_init.infos))

        return InfoGroup('Init Strategies', InfoList(W_rec_info, W_in_info, W_out_info, b_rec_info, b_out_info))
