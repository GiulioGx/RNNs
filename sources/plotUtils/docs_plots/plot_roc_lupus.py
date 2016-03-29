import matplotlib.pyplot as plt
import numpy
from sklearn import metrics
from sklearn.metrics import roc_auc_score

from Configs import Configs

files = ['/home/giulio/Dropbox/completed/LupusDataset/lupusAll_thr92/run_4/scores.npz', '/home/giulio/Dropbox/completed/LupusDataset/lupusVip7_thr92/run_4/scores.npz']
colors = ['m', 'b']
legends = ['all_feats', 'selected_feats']

i = 0
for f in files:

    print('Analizing file: {}'.format(f))
    npz = numpy.load(f)
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
    sensitivity = tpr # same as recall

    precision2  = tp / (tp +fp)
    print(p, n)

    print('num positives: {}, num negatives: {}'.format(p, n))
    print('ROC score: {:.2f}'.format(roc_score))

    # ROC plot
    plt.figure(1)
    plt.plot(fpr, tpr, colors[i], linewidth=2)
    plt.legend(legends, shadow=True, fancybox=True, loc=1)
    plt.xlabel('false positives rate')
    plt.ylabel('true positives rate')

    # RECALL-PRECISION plot
    plt.figure(2)
    # plt.plot(recall, precision, colors[i], linewidth=2)
    plt.plot(sensitivity, precision2, colors[i], linewidth=2)
    plt.legend(legends, shadow=True, fancybox=True, loc=1)
    plt.xlabel('recall')
    plt.ylabel('precision')

    # SPECIFICITY-SENSIBILITY plot
    plt.figure(3)
    plt.plot(specificity, sensitivity, colors[i], linewidth=2)
    plt.legend(legends, shadow=True, fancybox=True, loc=1)
    plt.ylabel('sensibility')
    plt.xlabel('specificity')

    i+=1

plt.show()
