from Configs import Configs
from Infos import Info
import numpy

__author__ = 'giulio'


class Statistics(object):

    def __init__(self, max_it, check_freq):
        self.__dictionary = {}
        self.__check_freq = check_freq
        self.__current_it = 0
        self.__actual_length = 0
        self.__elapsed_time = 0
        self.__m = numpy.ceil(max_it / check_freq) - 1

    def update(self, info: Info, it, elapsed_time):

        j = it / self.__check_freq
        self.__current_it = it
        self.__actual_length += 1
        self.__elapsed_time = elapsed_time

        for e in info.elements:
            if e.name not in self.__dictionary:
                self.__dictionary[e.name] = numpy.zeros((self.__m,), dtype=Configs.floatType)
            self.__dictionary[e.name][j] = e.value

    @property
    def elapsed_time(self):
        return self.__elapsed_time

    @property
    def dictionary(self):
        return self.__dictionary





