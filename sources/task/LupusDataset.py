import math
import numpy
from scipy.io import loadmat

from Configs import Configs
from Paths import Paths
from infos.InfoElement import SimpleDescription
from task.Batch import Batch
from task.Dataset import Dataset


class LupusDataset(Dataset):
    def __init__(self, mat_file: str, seed: int = Configs.seed):
        mat_obj = loadmat(mat_file)

        self.__num_min_visit = 3  # XXX 2 schould work
        positive_patients = mat_obj['pazientiPositivi']
        negative_patients = mat_obj['pazientiNegativi']

        # TODO leaveone out

        feature_struct = mat_obj['selectedFeatures']
        self.__features_names = self.__find_features_names(feature_struct)

        data = numpy.concatenate((positive_patients, negative_patients), axis=0)
        self.__features_normalizations = LupusDataset.__find_normalization_factors(self.__features_names, data)

        positives, self.__max_visits_pos = self.__process_patients(positive_patients, self.__features_names,
                                                                   self.__features_normalizations)
        self.__early_positives, self.__late_positives = LupusDataset.__split_positive(positives)

        self.__negatives, self.__max_visits_neg = self.__process_patients(negative_patients,
                                                                          self.__features_names,
                                                                          self.__features_normalizations)

        self.__rng = numpy.random.RandomState(seed)
        self.__n_in = len(self.__features_names)
        self.__n_out = 1

    def __find_features_names(self, features):
        names = []
        if isinstance(features, numpy.ndarray) or isinstance(features, numpy.void):
            for obj in features:
                names += self.__find_features_names(obj)
            return numpy.unique(names)
        elif isinstance(features, numpy.str_):
            return [features]
        else:
            raise TypeError('got type: {}, expected type is "numpy.str_"', type(features))

    @staticmethod
    def __find_normalization_factors(fetures_names, data):
        vals = dict()
        for f in fetures_names:
            data_f = data[f]
            vals[f] = (dict(min=min(data_f), max=max(data_f)))
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

    def __process_patients(self, mat_data, features_names, features_normalizations):

        patients_datas = []
        max_visits = 0

        n_features = len(features_names)
        patients_ids = numpy.unique(mat_data['PazienteId'])
        for id in patients_ids:
            visits_indexes = mat_data['PazienteId'] == id.item()
            visits = mat_data[visits_indexes]
            visits = sorted(visits, key=lambda visit: visit['numberVisit'].item())
            n_visits = len(visits)
            if n_visits >= self.__num_min_visit:
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

        exs = (self.__early_positives, self.__late_positives, self.__negatives)
        bs = int(math.ceil(float(batch_size) / len(exs)))

        indexes = []
        max_length = 0

        for e in exs:
            e_indexes = self.__rng.randint(size=(bs, 1), low=0, high=len(e))[0]
            indexes.append(e_indexes)
            max_length = max(len(e[max(e_indexes, key=lambda i: len(e[i]['targets']))]['targets']), max_length)

        return self.__build_batch(indexes, exs, max_length)
        #return self.train_set[0]

    @property
    def n_in(self):
        return self.__n_in

    @property
    def n_out(self):
        return self.__n_out

    @property
    def train_set(self):
        exs = (self.__early_positives, self.__late_positives, self.__negatives)
        indexes = []
        for e in exs:
            indexes.append(range(len(e)))
        max_length = max(self.__max_visits_neg, self.__max_visits_pos)
        return [self.__build_batch(indexes, exs, max_length)]

    @property
    def infos(self):
        description = []
        description.append('Lupus Dataset:\n')
        description.append('features: {}\n'.format(self.__features_names))
        description.append('normalizations: {}\n'.format(self.__features_normalizations))
        description.append('{} early positive patients found\n'.format(len(self.__early_positives)))
        description.append('{} late positive patients found\n'.format(len(self.__late_positives)))
        description.append('{} negative patients found\n'.format(len(self.__negatives)))

        return SimpleDescription(''.join(description))


if __name__ == '__main__':
    dataset = LupusDataset(Paths.lupus_path)
    print(dataset.infos)
    batch = dataset.get_train_batch(batch_size=3)
    print(str(batch))
