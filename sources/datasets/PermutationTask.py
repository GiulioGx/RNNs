
from Configs import Configs
import numpy
from datasets.Batch import Batch
from datasets.Task import Task
from infos.InfoElement import SimpleDescription, PrintableInfoElement
from infos.InfoList import InfoList


class PermutationTask(Task):
    def error_fnc(self, t, y, mask):
        return Batch.last_step_one_hot(t=t, y=y, mask=mask)

    @property
    def n_out(self):
        return self.__n_out

    @property
    def n_in(self):
        return self.__n_in

    def get_batch(self, batch_size: int) -> Batch:
        length = self.__rng.randint(int(self.__min_length * .1)) + self.__min_length

        # Copyright (c) 2012-2013, Razvan Pascanu
        # All rights reserved.
        randvals = self.__rng.randint(98, size=(length + 1, batch_size)) + 2
        val = self.__rng.randint(2, size=(batch_size,))
        randvals[numpy.zeros((batch_size,), dtype='int32'),
                 numpy.arange(batch_size)] = val
        randvals[numpy.ones((batch_size,), dtype='int32') * length,
                 numpy.arange(batch_size)] = val
        _targ = randvals[1:]
        _inp = randvals[:-1]
        inp = numpy.zeros((length, batch_size, 100), dtype=Configs.floatType)
        # targ = numpy.zeros((length, batchsize, 100), dtype=self.floatX)
        targ = numpy.zeros((1, batch_size, 100), dtype=Configs.floatType)
        inp.reshape((length * batch_size, 100))[ \
            numpy.arange(length * batch_size),
            _inp.flatten()] = 1.
        # targ.reshape((length*batchsize, 100))[\
        #        numpy.arange(batchsize),
        #        _targ[-1].flatten()] = 1.
        targ.reshape((batch_size, 100))[ \
            numpy.arange(batch_size),
            _targ[-1].flatten()] = 1.

        targ = targ.reshape((batch_size, 100))
        # return inp, targ

        ############
        # inputs = numpy.zeros((length, self.__n_in, batch_size), dtype=Configs.floatType)
        outputs = numpy.zeros((length, self.__n_out, batch_size), dtype=Configs.floatType)

        outputs[-1, :, :] = numpy.swapaxes(targ, 1, 0)
        inputs = numpy.swapaxes(inp, 2, 1)

        mask = numpy.zeros_like(outputs)
        mask[-1, :, :] = 1

        return Batch(inputs.astype(dtype=Configs.floatType), outputs.astype(Configs.floatType), mask)

    @property
    def infos(self):
        return InfoList(SimpleDescription('permutation'),
                        PrintableInfoElement('min_length', ':d', self.__min_length))

    def __init__(self, min_length: int, seed: int):
        self.__n_in = 100
        self.__n_out = 100
        self.__min_length = min_length
        self.__rng = numpy.random.RandomState(seed)


if __name__ == '__main__':
    seed = 13
    print('Testing Temporal Order datasets ...')
    task = PermutationTask(25, seed)
    batch = task.get_batch(3)
    print(str(batch))
