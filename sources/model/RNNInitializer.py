from Configs import Configs
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from infos.InfoProducer import SimpleInfoProducer
from initialization.GaussianInit import GaussianInit
from initialization.MatrixInit import MatrixInit


class RNNInitializer(SimpleInfoProducer):
    def __init__(self, W_rec_init: MatrixInit = GaussianInit(), W_in_init: MatrixInit = GaussianInit(),
                 W_out_init: MatrixInit = GaussianInit(), b_rec_init: MatrixInit = GaussianInit(),
                 b_out_init: MatrixInit = GaussianInit()):
        self.__W_rec_init = W_rec_init
        self.__W_in_init = W_in_init
        self.__W_out_init = W_out_init
        self.__b_rec_init = b_rec_init
        self.__b_out_init = b_out_init

    def generate_variables(self, n_in: int, n_out: int, n_hidden: int):
        # init network matrices
        W_rec = self.__W_rec_init.init_matrix((n_hidden, n_hidden), Configs.floatType)
        W_in = self.__W_in_init.init_matrix((n_hidden, n_in), Configs.floatType)
        W_out = self.__W_out_init.init_matrix((n_out, n_hidden), Configs.floatType)

        # init biases
        b_rec = self.__b_rec_init.init_matrix((n_hidden, 1), Configs.floatType)
        b_out = self.__b_out_init.init_matrix((n_out, 1), Configs.floatType)

        return W_rec, W_in, W_out, b_rec, b_out

    @property
    def infos(self):
        W_rec_info = InfoGroup('W_rec', InfoList(self.__W_rec_init.infos))
        W_in_info = InfoGroup('W_in', InfoList(self.__W_in_init.infos))
        W_out_info = InfoGroup('W_out', InfoList(self.__W_out_init.infos))
        b_rec_info = InfoGroup('b_rec', InfoList(self.__b_rec_init.infos))
        b_out_info = InfoGroup('b_out', InfoList(self.__b_out_init.infos))

        return InfoGroup('Init Strategies', InfoList(W_rec_info, W_in_info, W_out_info, b_rec_info, b_out_info))
