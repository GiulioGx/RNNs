from Tasks.Batch import Batch
import numpy
from configs import Configs

__author__ = 'giulio'


class MarkerBasedTask:

    def __init__(self, input_fnc, output_fnc, n_in, n_out, min_length, seed):

        self.__input_fnc = input_fnc
        self.__output_fnc = output_fnc
        self.__n_in = n_in
        self.__n_out = n_out
        self.__min_length = min_length
        self.__rng = numpy.random.RandomState(seed)

    def get_batch(self, batch_size: int):

        length = self.__rng.randint(int(self.__min_length * .1)) + self.__min_length

        # data init
        inputs = numpy.zeros((batch_size, length, self.__n_in), dtype=Configs.floatType)
        outputs = numpy.zeros((batch_size, length, self.__n_out), dtype=Configs.floatType)

        # marker positions
        p0 = self.__rng.randint(int(length * .1), size=(batch_size,))
        p1 = self.__rng.randint(int(length * .4), size=(batch_size,)) + int(length * .1)

        # markers value (channel 0)
        inputs[numpy.arange(batch_size), p0, numpy.zeros((batch_size,),
                                                         dtype='int32')] = 1
        inputs[numpy.arange(batch_size), p1, numpy.zeros((batch_size,),
                                                         dtype='int32')] = 1
        # random inputs (channel 1)
        inputs[:, :, 1] = self.__input_fnc(batch_size, length)

        # outputs
        self.__output_fnc(inputs, outputs, p0, p1)

        return Batch(inputs, outputs)

    @property
    def n_in(self):
        return self.__n_in

    @property
    def n_out(self):
        return self.__n_out
