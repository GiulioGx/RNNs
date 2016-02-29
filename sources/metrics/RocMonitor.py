import numpy
from sklearn.metrics import roc_auc_score

from infos.Info import Info
from infos.InfoElement import PrintableInfoElement
from metrics.MeasureMonitor import MeasureMonitor


class RocMonitor(MeasureMonitor): #TODO realvalue criterion
    def __init__(self):
        self.__value = numpy.inf

    @property
    def value(self):
        return self.__value

    def get_symbols(self, y, t, mask) -> list:
        return [y, t, mask]

    def update(self, measures: list):

        scores = []
        labels = []

        for measure in measures:
            y = measure[0]
            t = measure[1]
            mask = measure[2]
            m = sum(mask, 1)

            for i in range(y.shape[2]):
                idx = sum(m[:, i])
                y_filtered = y[0:idx, :, i]
                t_filtered = t[0:idx, :, i]

                s = sum(t_filtered)
                if s == 0:  # negatives
                    comparing_idx = numpy.argmax(y_filtered)
                elif s == len(t_filtered):  # early positives
                    comparing_idx = numpy.argmin(y_filtered)
                else:
                    comparing_idx = numpy.min(numpy.nonzero(y_filtered))

                scores.append(y_filtered[comparing_idx].item())
                labels.append(t_filtered[comparing_idx].item())
        self.__value = roc_auc_score(y_true=numpy.array(labels), y_score=numpy.array(scores))

    @property
    def info(self) -> Info:
        return PrintableInfoElement('roc_auc', ':2.2f', self.__value)
