import matplotlib.pyplot as plt
import numpy
from sklearn import metrics
from sklearn.metrics import roc_auc_score

from Configs import Configs

file = Configs.output_dir + 'Lupus_k/scores.npz'
npz = numpy.load(file)
scores = npz['scores']
labels = npz['labels']
fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
score = roc_auc_score(y_true=labels, y_score=scores)
print('ROC score: {:.2f}'.format(score))

plt.plot(fpr, tpr, 'm', linewidth=2)
plt.legend(['ROC\n (area under curve:{:.2f})'.format(score)], shadow=True, fancybox=True, loc=1)
plt.xlabel('false positives rate')
plt.ylabel('true positives rate')
plt.show()
