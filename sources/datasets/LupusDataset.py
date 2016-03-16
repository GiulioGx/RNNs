import abc
import math
from random import shuffle

import numpy
from scipy.io import loadmat

from Configs import Configs
from Paths import Paths
from infos.Info import Info, NullInfo
from infos.InfoElement import SimpleDescription, PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from datasets.Batch import Batch
from datasets.Dataset import Dataset


class BuildBatchStrategy(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def build_batch(self, patience):
        """build up a batch according to the strategy"""

    @abc.abstractmethod
    def keys(self) -> list:
        """return the keys of the sets to be used with this strategy"""


class PerVisitTargets(BuildBatchStrategy):
    def __init__(self):
        pass

    def keys(self) -> list:
        return ['early_pos', 'late_pos', 'neg']

    def build_batch(self, patience):
        feats = patience['features']
        targets = patience['targets']

        n_visits = len(targets)
        mask = numpy.ones_like(feats)
        return feats[0:n_visits - 1, :], targets[1:n_visits, :], mask


class PerPatienceTargets(BuildBatchStrategy):
    def keys(self) -> list:
        return ['neg', 'late_pos']

    def build_batch(self, patience):
        feats = patience['features']
        targets = patience['targets']

        non_zero_indexes = numpy.where(targets > 0)[0]

        if len(non_zero_indexes) > 0:
            first_positive_idx = numpy.min(non_zero_indexes)
            assert (first_positive_idx > 0)
            outputs = numpy.zeros(shape=(first_positive_idx, 1), dtype=Configs.floatType)
            outputs[-1] = 1
            inputs = feats[0:first_positive_idx, :]

        else:
            inputs = feats[0:-1, :]
            outputs = targets[0:-1, :]
        mask = numpy.zeros_like(outputs)
        mask[-1, :] = 1
        return inputs, outputs, mask


class LupusDataset(Dataset):
    num_min_visit = 2  # dicard patients with lass than visits
    num_min_visit_negative = 7  # discard negative patience with lass than visits

    @staticmethod
    def __load_mat(mat_file: str):
        mat_obj = loadmat(mat_file)

        positive_patients = mat_obj['pazientiPositivi']
        negative_patients = mat_obj['pazientiNegativi']

        features_struct = mat_obj['selectedFeatures']
        # features_struct = mat_obj['featuresVip7']
        features_names = LupusDataset.__find_features_names(features_struct)

        data = numpy.concatenate((positive_patients, negative_patients), axis=0)
        features_normalizations = LupusDataset.__find_normalization_factors(features_names, data)

        positives, max_visits_pos, pos_stats = LupusDataset.__process_patients(positive_patients, features_names,
                                                                               features_normalizations)
        early_positives, late_positives = LupusDataset.__split_positive(positives)
        shuffle(early_positives)
        shuffle(late_positives)

        negatives, max_visits_neg, neg_stats = LupusDataset.__process_patients(negative_patients,
                                                                               features_names,
                                                                               features_normalizations,
                                                                               LupusDataset.num_min_visit_negative)
        shuffle(negatives)

        description = ['Lupus Dataset:\n', 'features: {}\n'.format(features_names),
                       'normalizations: {}\n'.format(features_normalizations),
                       '{} early positive patients found\n'.format(len(early_positives)),
                       '{} late positive patients found\n'.format(len(late_positives)),
                       '{} negative patients found\n'.format(len(negatives)), 'positives stats:\n' + str(pos_stats),
                       'negatives stats:\n' + str(neg_stats)]

        infos = SimpleDescription(''.join(description))

        return early_positives, late_positives, negatives, max_visits_pos, max_visits_neg, features_names, infos

    @staticmethod
    def no_test_dataset(mat_file: str, strategy: BuildBatchStrategy = PerVisitTargets, seed: int = Configs.seed):
        early_positives, late_positives, negatives, max_visits_pos, max_visits_neg, features_names, infos = LupusDataset.__load_mat(
            mat_file)
        train_set = dict(early_pos=early_positives, late_pos=late_positives, neg=negatives, max_pos=max_visits_pos,
                         max_neg=max_visits_neg)
        data_dict = dict(train=train_set, test=train_set, features=features_names)
        return LupusDataset(data=data_dict, infos=infos, seed=seed, strategy=strategy)

    @staticmethod
    def __split_set(set, i, k):
        n = len(set)
        m = int(float(n) / k)
        start = m * i
        end = int(start + m if i < k - 1 else n)
        return set[0:start] + set[end:], set[start:end]

    @staticmethod
    def k_fold_test_datasets(mat_file: str, k: int = 1, strategy: BuildBatchStrategy = PerVisitTargets(),
                             seed: int = Configs.seed):
        early_positives, late_positives, negatives, max_visits_pos, max_visits_neg, features_names, infos = LupusDataset.__load_mat(
            mat_file)
        for i in range(k):
            eptr, epts = LupusDataset.__split_set(early_positives, i=i, k=k)
            lptr, lpts = LupusDataset.__split_set(late_positives, i=i, k=k)
            ntr, nts = LupusDataset.__split_set(negatives, i=i, k=k)

            train_set = dict(early_pos=eptr, late_pos=lptr, neg=ntr, max_pos=max_visits_pos,
                             max_neg=max_visits_neg)
            test_set = dict(early_pos=epts, late_pos=lpts, neg=nts, max_pos=max_visits_pos,
                            max_neg=max_visits_neg)
            data_dict = dict(train=train_set, test=test_set, features=features_names)
            yield LupusDataset(data=data_dict, infos=infos, seed=seed, strategy=strategy)

    @staticmethod
    def get_set_info(set):
        return InfoList(PrintableInfoElement('early_pos', ':d', len(set['early_pos'])),
                        PrintableInfoElement('late_pos', ':d', len(set['late_pos'])),
                        PrintableInfoElement('neg', ':d', len(set['neg'])))

    def __init__(self, data: dict, infos: Info = NullInfo(), strategy: BuildBatchStrategy = PerVisitTargets(),
                 seed: int = Configs.seed):

        self.__train = data['train']
        self.__test = data['test']
        self.__features = data['features']
        self.__rng = numpy.random.RandomState(seed)
        self.__n_in = len(self.__features)
        self.__n_out = 1
        self.__build_batch_strategy = strategy  # TODO add infos

        split_info = InfoGroup('split', InfoList(InfoGroup('train', InfoList(LupusDataset.get_set_info(self.__train))),
                                                 InfoGroup('test', InfoList(LupusDataset.get_set_info(self.__test)))))
        self.__infos = InfoList(infos, split_info)

    @staticmethod
    def __find_features_names(features):
        names = []
        if isinstance(features, numpy.ndarray) or isinstance(features, numpy.void):
            for obj in features:
                names.extend(LupusDataset.__find_features_names(obj))
            return numpy.unique(names)
        elif isinstance(features, numpy.str_):
            return [str(features)]
        else:
            raise TypeError('got type: {}, expected type is "numpy.str_"', type(features))

    @staticmethod
    def __find_normalization_factors(fetures_names, data):
        vals = dict()
        for f in fetures_names:
            data_f = data[f]
            vals[f] = (dict(min=min(data_f).item().item(), max=max(data_f).item().item()))
        return vals

    @staticmethod
    def __split_positive(positives):

        early_positives = []
        late_positives = []

        for p in positives:
            targets = p['targets']
            assert (sum(targets) > 0)
            if targets[0] > 0:
                early_positives.append(p)
            else:
                late_positives.append(p)

        return early_positives, late_positives

    @staticmethod
    def __process_patients(mat_data, features_names, features_normalizations, min_visits: int = num_min_visit):

        patients_datas = []
        max_visits = 0

        stats = LupusDataset.Stats()

        n_features = len(features_names)
        patients_ids = numpy.unique(mat_data['PazienteId'])
        for id in patients_ids:
            visits_indexes = mat_data['PazienteId'] == id.item()
            visits = mat_data[visits_indexes]
            visits = sorted(visits, key=lambda visit: visit['numberVisit'].item())
            stats.add_patience(visits)
            n_visits = len(visits)
            if n_visits >= min_visits:
                max_visits = max(max_visits, n_visits)
                pat_matrix = numpy.zeros(shape=(n_visits, n_features))
                target_vec = numpy.zeros(shape=(n_visits, 1))
                for j in range(n_visits):
                    target_vec[j] = 1 if visits[j]['sdi'] > 0 else 0  # sdi is greater than one for positive patients
                    for k in range(n_features):
                        f_name = features_names[k]
                        a = features_normalizations[f_name]['min']
                        b = features_normalizations[f_name]['max']
                        pat_matrix[j, k] = (visits[j][f_name].item() - a) / (b - a)
                patients_datas.append(dict(features=pat_matrix, targets=target_vec))

        return patients_datas, max_visits, stats

    @staticmethod
    def __print_patience(pat_dict):
        features = pat_dict['features']
        targets = pat_dict['targets']
        n_visits = len(targets)
        assert (n_visits == features.shape[0])
        for i in range(n_visits):
            print('Visit {}:\n features: {}\t targets(sdi): {}'.format(i, features[i, :], targets[i]))

    @staticmethod
    def print_results(patient_number, batch, y):
        n_visits = int(sum(sum(batch.mask[:, :, patient_number])).item())
        print('Patient {}: number of visits: {}'.format(patient_number, n_visits))
        print('Net output,\tLabel')
        for i in range(n_visits):
            print('\t{:01.2f},\t {:01.0f}'.format(y[i, :, patient_number].item(),
                                                  batch.outputs[i, :, patient_number].item()))

    def __build_batch(self, indexes, sets, max_length) -> Batch:
        max_length -= 1
        n_sets = len(sets)
        n_batch_examples = 0
        for i in indexes:
            n_batch_examples += len(i)
        inputs = numpy.zeros(shape=(max_length, self.__n_in, n_batch_examples))
        outputs = numpy.zeros(shape=(max_length, self.__n_out, n_batch_examples))
        mask = numpy.zeros_like(outputs)

        partial_idx = 0
        for i in range(n_sets):
            bs = len(indexes[i])
            for j in range(bs):
                idx = indexes[i][j]
                pat = sets[i][idx]

                feats, targets, pat_mask = self.__build_batch_strategy.build_batch(pat)
                n = feats.shape[0]
                assert (n == targets.shape[0])

                index = partial_idx + j
                inputs[0:n, :, index] = feats
                outputs[0:n, :, index] = targets
                mask[0:n, :, index] = pat_mask
            partial_idx += bs
        return Batch(inputs, outputs, mask)

    def __sets_from_keys(self, data):
        sets = []
        for key in self.__build_batch_strategy.keys():
            sets.append(data[key])
        return sets

    # def get_train_batch(self, batch_size:int):
    #     exs = self.__sets_from_keys(self.__train)
    #     pool = []
    #     for e in exs:
    #         pool += e
    #
    #     indexes = self.__rng.randint(size=(batch_size, 1), low=0, high=len(pool))
    #     max_length = len(pool[max(indexes, key=lambda i: len(pool[i]['targets']))]['targets'])
    #     return self.__build_batch([indexes], [pool], max_length)

    def get_train_batch(self, batch_size: int) -> Batch:
        """return a 'Batch' of size 'batch_size'"""
        exs = self.__sets_from_keys(self.__train)
        bs = int(math.ceil(float(batch_size) / len(exs)))

        indexes = []
        max_length = 0

        for e in exs:
            e_indexes = self.__rng.randint(size=(bs, 1), low=0, high=len(e))
            indexes.append(e_indexes)
            max_length = max(len(e[max(e_indexes, key=lambda i: len(e[i]['targets']))]['targets']), max_length)

        return self.__build_batch(indexes, exs, max_length)

    @property
    def n_in(self):
        return self.__n_in

    @property
    def n_out(self):
        return self.__n_out

    def __get_batch_from_whole_sets(self, sets: list, max_length: int) -> Batch:
        indexes = []
        for e in sets:
            indexes.append(range(len(e)))
        return self.__build_batch(indexes, sets, max_length)

    def __get_set(self, set, mode):
        max_length = max(set['max_pos'], set['max_neg'])
        if mode == 'whole':
            sets = self.__sets_from_keys(set)
            return [self.__get_batch_from_whole_sets(sets, max_length)]
        elif mode == 'split':
            keys = self.__build_batch_strategy.keys()
            splits = self.__sets_from_keys(set)
            assert (len(keys) == len(splits))
            d = dict()
            for i in range(len(keys)):
                d[keys[i]] = self.__get_batch_from_whole_sets([splits[i]], max_length=max_length)
            return d
        else:
            raise ValueError('unsupported value')  # TODO

    @property
    def test_set(self):
        return self.__get_set(self.__test, mode='whole')

    @property
    def train_set(self):
        return self.__get_set(self.__train, mode='whole')

    @property
    def split_train(self):
        return self.__get_set(self.__train, mode='split')

    @property
    def split_test(self):
        return self.__get_set(self.__test, mode='split')

    @staticmethod
    def correct_prediction(y):
        """correct the prediction in such a way that the probabilities are monotonic non decreasing"""

        max_val = 0.
        result_y = y
        for i in range(y.shape[0]):
            i_val = result_y[i]
            result_y[i] = max(i_val, max_val)
            max_val = max(max_val, i_val)
        return result_y

    @staticmethod
    def get_scores_visits(y, t, mask):

        n_examples = y.shape[2]
        n_visits_max = n_examples * y.shape[0]
        reduced_mask = numpy.sum(mask, axis=1)
        scores = numpy.zeros(shape=(n_visits_max, 1), dtype=Configs.floatType)
        labels = numpy.zeros_like(scores)

        visit_count = 0
        for i in range(n_examples):
            n_visits = sum(reduced_mask[:, i])
            y_filtered = y[0:n_visits, :, i]
            t_filtered = t[0:n_visits, :, i]

            scores[visit_count:visit_count + n_visits] = y_filtered
            labels[visit_count:visit_count + n_visits] = t_filtered
            visit_count += n_visits

        return scores[0:visit_count], labels[0:visit_count]

    @staticmethod
    def get_scores_patients(y, t, mask):

        if numpy.sum(numpy.sum(numpy.sum(t))) <= 0:
            print('t', t)
            print('t_size ', t.shape)
        assert (numpy.sum(numpy.sum(numpy.sum(t))) > 0)

        n_examples = y.shape[2]
        reduced_mask = numpy.sum(mask, axis=1)
        scores = numpy.zeros(shape=(n_examples, 1), dtype=Configs.floatType)
        labels = numpy.zeros_like(scores)

        for i in range(n_examples):
            non_zero_indexes = numpy.where(reduced_mask[:, i] > 0)[0]
            idx = numpy.min(non_zero_indexes)

            scores[i] = y[idx, :, i]
            labels[i] = t[idx, :, i]
        assert (numpy.sum(scores) > 0 and numpy.sum(scores) != len(scores))
        return scores, labels

    @staticmethod
    def get_scores(y, t, mask):

        n_examples = y.shape[2]
        reduced_mask = numpy.sum(mask, axis=1)
        scores = numpy.zeros(shape=(n_examples, 1), dtype=Configs.floatType)
        labels = numpy.zeros_like(scores)

        for i in range(n_examples):
            n_visits = sum(reduced_mask[:, i])
            y_filtered = y[0:n_visits, :, i]
            t_filtered = t[0:n_visits, :, i]

            non_zero_indexes = numpy.nonzero(t_filtered)[0]
            zero_indexes = numpy.nonzero(t_filtered < 1)[0]

            n_non_zero = non_zero_indexes.shape[0]
            n_zero = zero_indexes.shape[0]
            assert (n_zero + n_non_zero == n_visits)

            if n_non_zero > 0 and n_zero > 0 and numpy.min(y_filtered[non_zero_indexes]) < numpy.max(
                    y_filtered[zero_indexes]):
                # in this case the prediction is non consistent whatever the threshold is
                scores[i] = -1.
                labels[i] = 1
            else:
                # in this case the probability are consistent, hence we choose as score (and label)
                # for the patience that of the visit which has the farthest score from the label
                to_compare_index = []
                to_compare_values = []
                if n_non_zero > 0:
                    p1_index = numpy.argmin(y_filtered[non_zero_indexes])
                    p1 = 1. - y_filtered[p1_index]
                    to_compare_index.append(p1_index)
                    to_compare_values.append(p1)
                if n_zero > 0:
                    p2_index = numpy.argmax(y_filtered[zero_indexes])
                    p2 = y_filtered[p2_index]
                    to_compare_index.append(p2_index)
                    to_compare_values.append(p2)

                j = to_compare_index[numpy.argmin(to_compare_values).item()]
                scores[i] = y_filtered[j].item()
                labels[i] = t_filtered[j].item()

        return scores, labels

    @property
    def infos(self):
        return self.__infos

    class Stats(object):

        class Measure(object):
            def __init__(self, name: str):
                self.__min = numpy.inf
                self.__max = 0
                self.__acc = 0.
                self.__count = 0
                self.__name = name

            def add_sample(self, value):
                self.__max = max(self.__max, value)
                self.__min = min(self.__min, value)
                self.__acc += value
                self.__count += 1

            @property
            def value(self):
                d = dict()
                d['min_' + self.__name] = self.__min
                d['max_' + self.__name] = self.__max
                d['mean_' + self.__name] = self.__acc / self.__count
                return d

        def __init__(self):
            self.__visit_measure = LupusDataset.Stats.Measure('num_visits')
            self.__age_measure = LupusDataset.Stats.Measure('age_span')

        def add_patience(self, visits):
            if not visits[0]['sdi'] > 0:  # discard early positives
                n_visits = len(visits)
                self.__visit_measure.add_sample(n_visits)
                age_span = visits[-1]['age'] - visits[0]['age']
                self.__age_measure.add_sample(age_span)

        def __str__(self):
            d = self.__age_measure.value
            d.update(self.__visit_measure.value)
            return str(d)


if __name__ == '__main__':
    dataset = LupusDataset.no_test_dataset(Paths.lupus_path, seed=13, strategy=PerPatienceTargets())
    print(dataset.infos)
    batch = dataset.get_train_batch(batch_size=3)
    print(str(batch))
