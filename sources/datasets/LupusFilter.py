import abc
import os

import numpy

from Paths import Paths
import datasets
import matplotlib.pyplot as plt
from infos.InfoElement import PrintableInfoElement, SimpleDescription
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from infos.InfoProducer import SimpleInfoProducer

from natsort import natsorted, ns

def format_table(*results_dirs: str):
    lup_prefix = 'Lupus Dataset_'
    filter_prefix = lup_prefix + 'temporal span visits filter_'

    results_dir = results_dirs[0]
    table_entries = []
    subdir = next(os.walk(results_dir))[1]
    subdir = natsorted(subdir, key=lambda y: y.lower())
    for run_dir in subdir:
        npz_file = numpy.load(results_dir + run_dir + '/scores.npz')
        npz_file2 = numpy.load(results_dirs[1] + run_dir + '/scores.npz')

        d = dict(pos=npz_file[lup_prefix + 'late positives'],
                 neg=npz_file[lup_prefix + 'negatives'],
                 min_v_pos=npz_file[filter_prefix + 'min visits pos'],
                 min_v_neg=npz_file[filter_prefix + 'min visits neg'],
                 lower_span=npz_file[filter_prefix + 'min age span lower'],
                 upper_span=npz_file[filter_prefix + 'min age span upper'],
                 score=npz_file['cum_score'],
                 score2 = npz_file2['cum_score']
                 )
        table_entries.append(d)

    rows = []
    for e in table_entries:
        sep = " & "
        s = str(e['upper_span'].item()) + sep + str(e['lower_span'].item()) + sep + str(
            e['min_v_neg'].item()) + sep + "{:.2f}".format(e['score2'].item()) + \
            sep + "{:.2f}".format(e['score'].item()) + sep + str(e['pos'].item()) + sep + str(e['neg'].item()) + """\\\\""" + '\n'
        rows.append(s)
    result = "".join(rows)
    print(result)


class LupusStats(object):
    def __init__(self, mat_data):
        self.__n_visits = []
        self.__age_spans = []

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


class VisitsFilter(SimpleInfoProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def select_visits(self, visits):
        """return a list of visits acordind to some criterion"""


class TemporalSpanFilter(VisitsFilter):
    @property
    def infos(self):
        return InfoGroup("temporal span visits filter",
                         InfoList(PrintableInfoElement("min age span upper", ':.1', self.__min_age_span_upper),
                                  PrintableInfoElement("min age span lower", ':.1', self.__min_age_span_lower),
                                  PrintableInfoElement("min visits neg", ':d', self.__min_visits),
                                  PrintableInfoElement("min visits pos", ":d", self.__min_visits_pos)))

    def __init__(self, min_age_span_upper, min_age_span_lower, min_visits_neg: int = 2, min_visits_pos=1):
        self.__min_age_span_upper = float(min_age_span_upper)
        self.__min_visits = min_visits_neg
        self.__min_age_span_lower = float(min_age_span_lower)
        self.__min_visits_pos = min_visits_pos

    def select_visits(self, visits):
        targets = [v["sdi"].item() for v in visits]

        if visits[0]['sdi'] > 0:
            cut_index = -1
        elif sum(targets) > 0:
            cut_index = numpy.min(numpy.nonzero(targets))  # this is the first visits where the patience is positive
            if cut_index <= self.__min_visits_pos:
                cut_index = -1
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


class NullFIlter(VisitsFilter):
    @property
    def infos(self):
        return SimpleDescription("NullFilter")

    def select_visits(self, visits):
        return visits


if __name__ == '__main__':
    # positive_patients, negative_patients, _ = datasets.LupusDataset.parse_mat(mat_file=Paths.lupus_path)
    # mat_data = numpy.concatenate((positive_patients, negative_patients), axis=0)
    # stats = LupusStats(mat_data=mat_data)
    # stats.plot_hists()

    format_table('/home/giulio/Dropbox/completed/LupusDataset/lupusVip7_thr92/', '/home/giulio/Dropbox/completed/LupusDataset/lupusAll_thr92/')