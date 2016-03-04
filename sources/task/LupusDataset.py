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
from task.Batch import Batch
from task.Dataset import Dataset


class LupusDataset(Dataset):
    num_min_visit = 3  # XXX 2 schould work

    @staticmethod
    def __load_mat(mat_file: str):
        mat_obj = loadmat(mat_file)

        positive_patients = mat_obj['pazientiPositivi']
        negative_patients = mat_obj['pazientiNegativi']

        feature_struct = mat_obj['selectedFeatures']
        features_names = LupusDataset.__find_features_names(feature_struct)

        data = numpy.concatenate((positive_patients, negative_patients), axis=0)
        features_normalizations = LupusDataset.__find_normalization_factors(features_names, data)

        positives, max_visits_pos = LupusDataset.__process_patients(positive_patients, features_names,
                                                                    features_normalizations)
        early_positives, late_positives = LupusDataset.__split_positive(positives)
        shuffle(early_positives)
        shuffle(late_positives)

        negatives, max_visits_neg = LupusDataset.__process_patients(negative_patients,
                                                                    features_names,
                                                                    features_normalizations)
        shuffle(negatives)

        description = ['Lupus Dataset:\n', 'features: {}\n'.format(features_names),
                       'normalizations: {}\n'.format(features_normalizations),
                       '{} early positive patients found\n'.format(len(early_positives)),
                       '{} late positive patients found\n'.format(len(late_positives)),
                       '{} negative patients found\n'.format(len(negatives))]

        infos = SimpleDescription(''.join(description))

        return early_positives, late_positives, negatives, max_visits_pos, max_visits_neg, features_names, infos

    @staticmethod
    def no_test_dataset(mat_file: str, seed: int = Configs.seed):
        early_positives, late_positives, negatives, max_visits_pos, max_visits_neg, features_names, infos = LupusDataset.__load_mat(
            mat_file)
        train_set = dict(early_pos=early_positives, late_pos=late_positives, neg=negatives, max_pos=max_visits_pos,
                         max_neg=max_visits_neg)
        data_dict = dict(train=train_set, test=train_set, features=features_names)
        return LupusDataset(data=data_dict, infos=infos, seed=seed)

    @staticmethod
    def __split_set(set, i, k):
        n = len(set)
        m = int(float(n) / k)
        start = m * i
        end = int(start + m if i < k - 1 else n)
        return set[start:end], set[0:start] + set[end:]

    @staticmethod
    def k_fold_test_datasets(mat_file: str, k: int = 1, seed: int = Configs.seed):
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
            yield LupusDataset(data=data_dict, infos=infos, seed=seed)

    @staticmethod
    def get_set_info(set):
        return InfoList(PrintableInfoElement('early_pos', ':d', len(set['early_pos'])),
                        PrintableInfoElement('late_pos', ':d', len(set['late_pos'])),
                        PrintableInfoElement('neg', ':d', len(set['neg'])))

    def __init__(self, data: dict, infos: Info = NullInfo(), seed: int = Configs.seed):

        self.__train = data['train']
        self.__test = data['test']
        self.__features = data['features']
        self.__rng = numpy.random.RandomState(seed)
        self.__n_in = len(self.__features)
        self.__n_out = 1

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
    def __process_patients(mat_data, features_names, features_normalizations):

        patients_datas = []
        max_visits = 0

        n_features = len(features_names)
        patients_ids = numpy.unique(mat_data['PazienteId'])
        for id in patients_ids:
            visits_indexes = mat_data['PazienteId'] == id.item()
            visits = mat_data[visits_indexes]
            visits = sorted(visits, key=lambda visit: visit['numberVisit'].item())
            n_visits = len(visits)
            if n_visits >= LupusDataset.num_min_visit:
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

        return patients_datas, max_visits

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

    def __build_batch(self, indexes, sets, max_length):
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
                feats = pat['features']
                targets = pat['targets']
                n_visits = len(targets)
                index = partial_idx + j
                inputs[0:n_visits - 1, :, index] = feats[0:n_visits - 1, :]
                outputs[0:n_visits - 1, :, index] = targets[1:n_visits]
                mask[0:n_visits - 1, :, index] = 1
            partial_idx += bs
        return Batch(inputs, outputs, mask)

    def get_train_batch(self, batch_size: int):

        exs = (self.__train['early_pos'], self.__train['late_pos'], self.__train['neg'])
        bs = int(math.ceil(float(batch_size) / len(exs)))

        indexes = []
        max_length = 0

        for e in exs:
            e_indexes = self.__rng.randint(size=(bs, 1), low=0, high=len(e))[0]
            indexes.append(e_indexes)
            max_length = max(len(e[max(e_indexes, key=lambda i: len(e[i]['targets']))]['targets']), max_length)

        return self.__build_batch(indexes, exs, max_length)

    @property
    def n_in(self):
        return self.__n_in

    @property
    def n_out(self):
        return self.__n_out

    def __get_train_batch_from_whole_sets(self, sets: tuple, max_length: int):
        indexes = []
        for e in sets:
            indexes.append(range(len(e)))
        return self.__build_batch(indexes, sets, max_length)

    def __get_set(self, set, mode):
        if mode == 'whole':
            sets = (set['early_pos'], set['late_pos'], set['neg'])
            max_length = max(set['max_pos'], set['max_neg'])
            return [self.__get_train_batch_from_whole_sets(sets, max_length)]
        elif mode == 'split':
            return dict(early_pos=self.__get_train_batch_from_whole_sets((set['early_pos'],), set['max_pos']),
                        late_pos=self.__get_train_batch_from_whole_sets((set['late_pos'],), set['max_pos']),
                        neg=self.__get_train_batch_from_whole_sets((set['neg'],), set['max_neg']))
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
    def get_scores(y, t, mask):

        n_examples = y.shape[2]
        reduced_mask = sum(mask, 1)
        scores = numpy.zeros(shape=(n_examples, 1), dtype=Configs.floatType)
        labels = numpy.zeros_like(scores)

        for i in range(y.shape[2]):
            idx = sum(reduced_mask[:, i])
            y_filtered = y[0:idx, :, i]
            t_filtered = t[0:idx, :, i]

            non_zero_indexes = numpy.nonzero(t_filtered)[0]
            zero_indexes = numpy.nonzero(t_filtered < 1)[0]

            assert (non_zero_indexes.shape[0] + zero_indexes.shape[0] == y_filtered.shape[0])

            to_compare_index = []
            to_compare_values = []
            if non_zero_indexes.shape[0] > 0:
                p1_index = numpy.argmin(y_filtered[non_zero_indexes])
                p1 = 1. - y_filtered[p1_index]
                to_compare_index.append(p1_index)
                to_compare_values.append(p1)
            if zero_indexes.shape[0] > 0:
                p2_index = numpy.argmax(y_filtered[zero_indexes])
                p2 = y_filtered[p2_index]
                to_compare_index.append(p2_index)
                to_compare_values.append(p2)

            j = to_compare_index[numpy.argmin(to_compare_values).item()]
            scores[i] = y_filtered[j].item()
            labels[i] = t_filtered[j].item()


            # s = sum(t_filtered)
            # if s == 0:  # negatives
            #     comparing_idx = numpy.argmax(y_filtered)
            # elif s == len(t_filtered):  # early positives
            #     comparing_idx = numpy.argmin(y_filtered)
            # else: # late positives
            #     comparing_idx = numpy.min(numpy.nonzero(y_filtered))
            # scores[i] = y_filtered[comparing_idx].item()
            # labels[i] = t_filtered[comparing_idx].item()
        return scores, labels

    @property
    def infos(self):
        return self.__infos


if __name__ == '__main__':
    dataset = LupusDataset.no_test_dataset(Paths.lupus_path)
    print(dataset.infos)
    batch = dataset.get_train_batch(batch_size=3)
    print(str(batch))
