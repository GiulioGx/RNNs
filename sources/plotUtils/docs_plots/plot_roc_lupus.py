import matplotlib.pyplot as plt
import numpy
from sklearn import metrics
from sklearn.metrics import roc_auc_score

from Configs import Configs

file = Configs.output_dir + 'Lupus_k/scores.npz'
npz = numpy.load(file)
scores = npz['scores']
labels = npz['labels']
fpr, tpr, thresholds_1 = metrics.roc_curve(labels, scores, pos_label=1)
roc_score = roc_auc_score(y_true=labels, y_score=scores)
precision, recall, thresholds_2 = metrics.precision_recall_curve(labels, scores)

# print('t', thresholds_1-thresholds_2)

p = len(numpy.nonzero(labels)[0])  # number of positives
n = len(labels) - p  # number of negatives
fp = fpr * p  # false positves
tp = tpr * p  # true positives
tn = n - fp  # true negatives
fn = p - tp  # false negatives

specificity = tn / n
sensibility = tpr # same as recall

print('num positives: {}, num negatives: {}'.format(p, n))

print('ROC score: {:.2f}'.format(roc_score))

# ROC plot
plt.figure(1)
plt.plot(fpr, tpr, 'm', linewidth=2)
plt.legend(['ROC\n (area under curve:{:.2f})'.format(roc_score)], shadow=True, fancybox=True, loc=1)
plt.xlabel('false positives rate')
plt.ylabel('true positives rate')

# RECALL-PRECISION plot
plt.figure(2)
plt.plot(recall, precision, 'm', linewidth=2)
# plt.legend([format(roc_score)], shadow=True, fancybox=True, loc=1)
plt.xlabel('recall')
plt.ylabel('precision')

# SPECIFICITY-SENSIBILITY plot
plt.figure(3)
plt.plot(specificity, sensibility, 'm', linewidth=2)
# plt.legend([format(roc_score)], shadow=True, fancybox=True, loc=1)
plt.ylabel('sensibility')
plt.xlabel('specificity')

plt.show()
