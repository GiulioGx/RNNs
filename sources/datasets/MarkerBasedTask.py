from datasets.Batch import Batch
import numpy
from Configs import Configs

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
        inputs = numpy.zeros((length, self.__n_in, batch_size), dtype=Configs.floatType)
        outputs = numpy.zeros((length, self.__n_out, batch_size), dtype=Configs.floatType)

        # marker positions
        p0 = self.__rng.randint(int(length * .1), size=(batch_size,))
        p1 = self.__rng.randint(int(length * .4), size=(batch_size,)) + int(length * .1)

        # markers value (channel 0)
        inputs[p0, numpy.zeros((batch_size,), dtype='int32'), numpy.arange(batch_size)] = 1
        inputs[p1, numpy.zeros((batch_size,), dtype='int32'), numpy.arange(batch_size)] = 1
        # random inputs (channel 1)
        inputs[:, 1, :] = self.__input_fnc(batch_size, length)  # FIXME 1

        mask = numpy.zeros_like(outputs)
        mask[-1, :, :] = 1

        # outputs
        self.__output_fnc(inputs, outputs, p0, p1)

        return Batch(inputs, outputs, mask)

    @property
    def n_in(self):
        return self.__n_in

    @property
    def n_out(self):
        return self.__n_out
