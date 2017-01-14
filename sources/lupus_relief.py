from Paths import Paths
from datasets.LupusDataset import LupusDataset, PerPatienceTargets
import numpy
from datasets.LupusFilter import TemporalSpanFilter
from selfea.datasets.CustomDataset import CustomDataset
from selfea.rankers.ReliefF import ReliefF


def convert_X(X, mask):
    # output = numpy.zeros(shape=(X.shape[2], X.shape[1]))
    output_list = []

    for i in range(X.shape[2]):
        l = numpy.argmax(mask[:, 0, i])
        output_list.append(X[:l + 1, :, i])
    return numpy.concatenate(output_list)


# def convert_Y(y, mask):
#     # output = numpy.zeros_like(mask)
#     output_list = []
#     for i in range(mask.shape[2]):
#         l = numpy.argmax(mask[:, 0, i])
#         output_list.append(output[l, 0, i] = y[i]
#     return output


min_age_lower = 0.8  # 0.8, 1, 2]
min_age_upper = 0.8  # [0.8, 1, 2]
min_visits_neg = 5  # [1, 2, 3, 4, 5]
min_visits_pos = 1
strategy = PerPatienceTargets()

dataset = LupusDataset.no_test_dataset(Paths.lupus_path, strategy=strategy,
                                       visit_selector=TemporalSpanFilter(min_age_span_upper=min_age_upper,
                                                                         min_age_span_lower=min_age_lower,
                                                                         min_visits_neg=min_visits_neg,
                                                                         min_visits_pos=min_visits_pos))

train_set = dataset.train_set[0]
X = convert_X(train_set.inputs, train_set.mask)
y = convert_X(train_set.outputs, train_set.mask)

# print(y)

relief = ReliefF(k=10, sigma=2)

ranking = relief.rank(CustomDataset(X, y))

# print(ranking)
#print(dataset.infos)

features_names = ['APS', 'DNA', 'FM', 'Hashimoto', 'MyasteniaGravis', 'SdS',
                  'arterialthrombosis', 'arthritis', 'c3level', 'c4level', 'dislipidemia', 'hcv',
                  'hematological', 'hypertension', 'hypothyroidism', 'kidney', 'mthfr', 'npsle',
                  'pregnancypathology', 'serositis', 'sex', 'skinrash', 'sledai2kInferred',
                  'venousthrombosis']

order = numpy.argsort(ranking)

for i in order:
    print(features_names[i], ranking[i])

