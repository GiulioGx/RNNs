import numpy
from sklearn.metrics import roc_auc_score

from infos.Info import Info
from infos.InfoElement import PrintableInfoElement
from metrics.RealValuedMonitor import RealValuedMonitor


class RocMonitor(RealValuedMonitor):
    def __init__(self, score_fnc):
        super().__init__(0)
        self.__score_fnc = score_fnc

    def get_symbols(self, y, t, mask) -> list:
        return [y, t, mask]

    def update(self, measures: list):
        scores_list = []
        labels_list = []

        for measure in measures:
            y = measure[0]
            t = measure[1]
            mask = measure[2]

            scores, labels = self.__score_fnc(y, t, mask)
            scores_list.append(scores)
            labels_list.append(labels)

        self._current_value = roc_auc_score(y_true=numpy.concatenate(*labels_list, axis=0),
                                            y_score=numpy.concatenate(*scores_list, axis=0))

    @property
    def info(self) -> Info:
        return PrintableInfoElement('roc_auc', ':2.2f', self._current_value)
