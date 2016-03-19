import abc

import numpy

from Paths import Paths
import datasets
import matplotlib.pyplot as plt

from infos.InfoElement import PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from infos.InfoProducer import SimpleInfoProducer


class LupusFilter(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def is_to_discard(self, visits) -> bool:
        """return True or False whether the patience schould be be discarde or not"""


class NullFilter(LupusFilter):
    def is_to_discard(self, visits) -> bool:
        return False


class TemporalSpanFilter(LupusFilter):
    def __init__(self, min_age_span, min_visits_before_status_change: int = 2):
        self.__min_age_span = min_age_span
        self.__min_visits_before_status_change = min_visits_before_status_change

    def is_to_discard(self, visits) -> bool:

        if visits[0]["sdi"] > 0:
            return True
        elif sum([v["sdi"].item() for v in visits]) > 0:
            return False
        else:

            last_sdi = 1 if visits[-1]["sdi"].item() > 0 else 0
            count = len(visits) - 1
            last_visit_age = visits[-1]["age"].item()
            sdi_change_found = False
            age_span = 0
            while age_span < self.__min_age_span and count >= 0 and not sdi_change_found:
                curr_sdi = 1 if visits[count]["sdi"].item() > 0 else 0
                if curr_sdi != last_sdi:
                    sdi_change_found = True
                age_span = last_visit_age - visits[count]["age"].item()
                count -= 1
        return age_span < self.__min_age_span or count <= self.__min_visits_before_status_change


class MinVisitsFilter(LupusFilter):
    def __init__(self, n):
        self.__n = n

    def is_to_discard(self, visits) -> bool:
        return len(visits) < self.__n


class AggregateFilter(LupusFilter):
    def is_to_discard(self, visits) -> bool:
        discard = False
        count = 0
        while not discard and count < len(filter):
            discard = discard or self.__filters[count]
            count += 1
        return discard

    def __init__(self, *filters: LupusFilter):
        self.__filters = filters


class LupusStats(object):
    def __init__(self, mat_file: str):
        positive_patients, negative_patients, features_names = datasets.LupusDataset.parse_mat(mat_file=mat_file)
        mat_data = numpy.concatenate((positive_patients, negative_patients), axis=0)
        mat_data = positive_patients

        self.__n_visits = []
        self.__age_spans = []

        n_features = len(features_names)
        patients_ids = numpy.unique(mat_data['PazienteId'])
        for id in patients_ids:
            visits_indexes = mat_data['PazienteId'] == id.item()
            visits = mat_data[visits_indexes]
            visits = sorted(visits, key=lambda visit: visit['numberVisit'].item())

            if visits[0]["sdi"] <= 0:
                cutoff = numpy.min(numpy.nonzero([v["sdi"].item() for v in visits]))
                self.__n_visits.append(cutoff)
                ages = [v["age"].item() for v in visits]

                age_span = max(ages) - min(ages)
                self.__age_spans.append(age_span)

    def plot_hists(self):
        plt.figure(1)
        plt.hist(self.__n_visits)
        plt.xlabel("num visits")
        plt.ylabel("num patients")

        print('Num Visitis-> min: {}, max: {}, mean: {}'.format(min(self.__n_visits), max(self.__n_visits),
                                                                numpy.mean(self.__n_visits)))

        plt.figure(2)
        print(self.__age_spans)
        plt.hist(self.__age_spans, bins=20)
        plt.xlabel("age span (years)")
        plt.ylabel("num patients")

        print('Age span (years)-> min: {:.2f}, max: {:.2f}, mean: {:.2f}'.format(min(self.__age_spans),
                                                                                 max(self.__age_spans),
                                                                                 numpy.mean(self.__age_spans)))

        plt.show()


class VisitsSelector(SimpleInfoProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def select_visits(self, visits):
        """return a list of visits acordind to some criterion"""


class TemporalSpanSelector(VisitsSelector):
    @property
    def infos(self):
        return InfoGroup("temporal span visits filter",
                         InfoList(PrintableInfoElement("min age span upper", ':.1', self.__min_age_span_upper),
                                  PrintableInfoElement("min age span lower", ':.1', self.__min_age_span_lower),
                                  PrintableInfoElement("min visits", ':d', self.__min_visits)))

    def __init__(self, min_age_span_upper, min_age_span_lower, min_visits: int = 2):
        self.__min_age_span_upper = min_age_span_upper
        self.__min_visits = min_visits
        self.__min_age_span_lower = min_age_span_lower

    def select_visits(self, visits):
        targets = [v["sdi"].item() for v in visits]

        if visits[0]['sdi'] > 0:
            cut_index = -1
        elif sum(targets) > 0:
            cut_index = numpy.min(numpy.nonzero(targets))  # this is the first visits where the patience is positive
        else:  # this is the case of always negative patients
            n_visits = len(visits)
            cut_index = n_visits
            prev_age = visits[-1]["age"]
            age_span = 0
            while cut_index > self.__min_visits and age_span < self.__min_age_span_upper:
                age_span = prev_age - visits[cut_index - 1]["age"]
                cut_index -= 1

            lower_age_span = visits[0 if (cut_index < 0 or cut_index >= len(visits)) else cut_index]["age"] - visits[0][
                "age"]
            if cut_index < self.__min_visits or age_span < self.__min_age_span_upper or lower_age_span < self.__min_age_span_lower:
                cut_index = -1

        return visits[0:cut_index + 1]


class NullSelector(VisitsSelector):
    def select_visits(self, visits):
        return visits


if __name__ == '__main__':
    stats = LupusStats(Paths.lupus_path)
    stats.plot_hists()
