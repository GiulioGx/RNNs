from sklearn import metrics
from sklearn.metrics import roc_auc_score

from Configs import Configs
from Paths import Paths
from model import RNN
from datasets.LupusDataset import LupusDataset
import matplotlib.pyplot as plt

file = Configs.output_dir + 'Lupus3/best_model.npz'
net = RNN.load_model(file)
dataset = LupusDataset.no_test_dataset(Paths.lupus_path)

print('---Late positive patient example ---')
batch = dataset.split_train['late_pos']
y = net.net_ouput_numpy(batch.inputs)[0]
patient_number = 12
LupusDataset.print_results(patient_number, batch, y)

print('\n---Early positive patient example ---')
batch = dataset.split_train['early_pos']
y = net.net_ouput_numpy(batch.inputs)[0]
patient_number = 3
LupusDataset.print_results(patient_number, batch, y)

print('\n---Negative patient example ---')
batch = dataset.split_train['neg']
y = net.net_ouput_numpy(batch.inputs)[0]
patient_number = 7
LupusDataset.print_results(patient_number, batch, y)

print('\n---Trainingset ROC---')
batch = dataset.train_set[0]
y = net.net_ouput_numpy(batch.inputs)[0]
scores, labels = dataset.get_scores(y, batch.outputs, batch.mask)
fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
score = roc_auc_score(y_true=labels, y_score=scores)
print('ROC score: {:.2f}'.format(score))

plt.plot(fpr, tpr, 'm', linewidth=2)
plt.legend(['ROC\n (area under curve:{:.2f})'.format(score)], shadow=True, fancybox=True, loc=1)
plt.xlabel('false positives rate')
plt.ylabel('true positives rate')
plt.show()
