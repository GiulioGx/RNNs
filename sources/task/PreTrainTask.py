import abc

from infos.SimpleInfoProducer import SimpleInfoProducer
from task.Task import Task

__author__ = 'giulio'


class PreTrainTask(Task):

    def __init__(self, task: Task):
        self.__task = task

    def get_batch(self, batch_size: int):

        batch = self.__task.get_batch(batch_size)

        return

    def error_fnc(self, t, y):
        return (((t[:, :, :] - y[:, :, :]) ** 2).sum(axis=1) > .04).mean().mean()

    def n_in(self):
        return self.__task.n_in

    def n_out(self):
        return self.__task.n_out
