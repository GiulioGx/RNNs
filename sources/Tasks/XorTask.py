import numpy
from Tasks.Batch import Batch

__author__ = 'giulio'


class XorTask:
    def __init__(self, min_length: int, seed: int):
        self.__min_length = min_length
        self.__n_in = 2
        self.__n_out = 1
        self.__rng = numpy.random.RandomState(seed)

    def get_batch(self, batch_size: int):
        length = self.__rng.randint(int(self.__min_length * .1)) + self.__min_length

        # XXX float32
        inputs = numpy.zeros((batch_size, length, self.__n_in), dtype='float32')
        outputs = numpy.zeros((batch_size, length, self.__n_out), dtype='float32')

        print(int(length * .1))
        print(int(length * .4))
        p0 = self.__rng.randint(int(length * .1), size=(batch_size,))
        p1 = self.__rng.randint(int(length * .4), size=(batch_size,)) + int(length * .1)

        # markers
        inputs[numpy.arange(batch_size), p0, numpy.zeros((batch_size,),
                                                         dtype='int32')] = 1
        inputs[numpy.arange(batch_size), p1, numpy.zeros((batch_size,),
                                                         dtype='int32')] = 1
        # binary inputs
        inputs[:, :, 1] = self.__rng.random_integers(0, 1, size=(batch_size, length)).astype('float32')

        return Batch(inputs, outputs)


if __name__ == '__main__':
    seed = 8
    print('Testing XOR task ...')
    task = XorTask(15, seed)
    batch = task.get_batch(3)
    print(str(batch))
