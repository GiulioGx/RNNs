__author__ = 'giulio'

import numpy


class XORTask:
    def __init__(self, min_length: int, seed: int):
        self.__min_length = min_length
        self.__n_in = 2
        self.__n_out = 1
        self.__rng = numpy.random.RandomState(seed)

    def get_batch(self, batch_size: int):
        length = self.__rng.randint(int(self.__min_length * .1)) + self.__min_length

        data = zeros((length, batch_size, size[0]), dtype=self.floatX)
        targ = zeros((batch_size, self.__n_out), dtype=self.floatX)


if __name__ == '__main__':
    print('Testing XOR task ...')
