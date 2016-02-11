import abc

import numpy

from infos.InfoElement import SimpleDescription, PrintableInfoElement
from infos.InfoList import InfoList
from infos.InfoProducer import SimpleInfoProducer
from initialization.RNNVarsInitializer import RNNVarsInitializer

__author__ = 'giulio'


class RNNProducer(SimpleInfoProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def generate_variables(self, n_in: int, n_out: int):
        """generate W_rec, W_in, W_out, b_rec, b_out in some way"""


class RNNLoader(RNNProducer):
    @property
    def infos(self):
        return SimpleDescription("RNN loading from file '{}'".format(self.__filename))

    def __init__(self, filename: str):
        self.__filename = filename

    def generate_variables(self, n_in: int, n_out: int):
        npz = numpy.load(self.__filename)

        W_rec = npz["W_rec"]
        W_in = npz["W_in"]
        W_out = npz["W_out"]
        b_rec = npz["b_rec"]
        b_out = npz["b_out"]

        if W_in.shape[1] != n_in or W_out.shape[0] != n_out:
            raise ValueError('the model specified does not have the number of units needed..')  # FOXME

        # # TODO pickel variable initializer
        #
        # filename, file_extension = os.path.splitext(filename)
        # pickle_file = filename + '.pkl'
        # activation_fnc_pkl, output_fnc_pkl = pickle.load(open(pickle_file, 'rb'))
        # if activation_fnc is None:
        #     activation_fnc = activation_fnc_pkl
        # if output_fnc is None:
        #     output_fnc = output_fnc_pkl

        return W_rec, W_in, W_out, b_rec, b_out


class RNNInitializer(RNNProducer):
    def generate_variables(self, n_in: int, n_out: int):
        return self.__variables_initializer.generate_variables(n_in, n_out, self.__n_hidden)

    @property
    def infos(self):
        return InfoList(PrintableInfoElement('n_hidden', '', self.__n_hidden), self.__variables_initializer.infos)

    def __init__(self, variables_initializer: RNNVarsInitializer, n_hidden: int = 100):
        self.__n_hidden = n_hidden
        self.__variables_initializer = variables_initializer
