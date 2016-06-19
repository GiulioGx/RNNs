import matplotlib.pyplot as plt
import numpy
import sys
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import os
#from Configs import Configs
from plotUtils.plt_utils import save_multiple_formats

files = ['/home/giulio/tmpLupus100_thr01_var/run_0/scores.npz']
colors = ['m', 'b']
legends = ['new feats']
size=(6,5)

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

    print(precision)
    print(recall)


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

    filename = sys.argv[0]
    filename = os.path.splitext(filename)[0]

    # ROC plot
    plt.figure(1, figsize=size)
    plt.plot(fpr, tpr, colors[i], linewidth=2)
    # plt.legend(legends, shadow=True, fancybox=True, loc=1)
    plt.xlabel('false positives rate')
    plt.ylabel('true positives rate')

    #save_multiple_formats(filename+'_roc')

    # RECALL-PRECISION plot
    plt.figure(2,figsize=size)
    plt.plot(recall, precision, colors[i], linewidth=2)
    # plt.plot(sensitivity, precision2, colors[i], linewidth=2)
    # plt.legend(legends, shadow=True, fancybox=True, loc=1)
    plt.xlabel('recall')
    plt.ylabel('precision')

    #save_multiple_formats(filename+'_recall_precision')


    # SPECIFICITY-SENSITIVITY plot
    plt.figure(3, figsize=size)
    plt.plot(specificity, sensitivity, colors[i], linewidth=2)
    # plt.legend(legends, shadow=True, fancybox=True, loc=1)
    plt.ylabel('sensitivity')
    plt.xlabel('specificity')

    #save_multiple_formats(filename+'_specificity_sensitivity')

    i+=1

plt.show()
