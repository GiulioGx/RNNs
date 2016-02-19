from infos.InfoElement import PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from metrics.Criterion import Criterion
from metrics.RealValuedMonitor import RealValuedMonitor


class ThresholdCriterion(Criterion):
    """this criterion is satisied if the monitored value is less than or equal to 'threshold'"""

    @property
    def infos(self):
        return InfoGroup('stopping criterion', InfoList(PrintableInfoElement('threshold', ':2.2f', self.__threshold)))

    def __init__(self, monitor: RealValuedMonitor, threshold: float):
        self.__threshold = threshold
        self.__monitor = monitor

    def is_satisfied(self) -> bool:
        return self.__monitor.value <= self.__threshold
