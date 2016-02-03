import abc

from infos.InfoElement import SimpleDescription, PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from infos.InfoProducer import SimpleInfoProducer
from task.Batch import Batch
from task.Task import Task
import numpy

from task.XorTaskHot import XorTaskHot

__author__ = 'giulio'


class PreTrainTask(Task): # XXX per ora funziona solo se hanno le stesse unità di output e di input
    def __init__(self, task: Task):
        self.__task = task

    def get_batch(self, batch_size: int):
        batch = self.__task.get_batch(batch_size)
        new_targets = numpy.zeros_like(batch.inputs)

        prev_values = numpy.zeros_like(new_targets[0, :, :])
        for i in range(new_targets.shape[0]):
            new_targets[i, :, :] = (prev_values + batch.inputs[i, :, :]).astype('float32') / 2
            prev_values = new_targets[i, :, :]

        return Batch(batch.inputs, new_targets)

    def error_fnc(self, t, y):  # not really relevant for this task
        return ((abs(t[:, :, :] - y[:, :, :])).sum(axis=0).sum(axis=0) > .04).mean()

    @property
    def n_in(self):
        return self.__task.n_in

    @property
    def n_out(self):
        return self.__task.n_in

    @property
    def infos(self):
        return InfoGroup('pretrain_task', self.__task.infos)


if __name__ == '__main__':
    seed = 13
    print('Testing Pretrain task ...')
    orig_task = XorTaskHot(144, seed)
    task = PreTrainTask(orig_task)
    batch = task.get_batch(1)
    print(str(batch))
