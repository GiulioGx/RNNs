import matplotlib.pyplot as plt
import numpy
import sys
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import os
import matplotlib.cm as cm

# from Configs import Configs
from plotUtils.plt_utils import save_multiple_formats

root = '/home/giulio/Lupus_model_selection/'
files = [root+'run_0/scores.npz',
         root+'run_1/scores.npz',
         root+'run_2/scores.npz',
         root+'run_3/scores.npz',
         root+'run_4/scores.npz',
         root+'run_5/scores.npz',
         root+'run_6/scores.npz',
         root+'run_7/scores.npz',
         root+'run_8/scores.npz']
colors = ['m', 'b', 'r', 'y', 'g', 'c', 'm', 'r', 'k']
legends = ['25h,  0.90thr', '50h,  0.90thr', '100h, 0.90thr', '25h,  0.95thr', '50h,  0.95thr', '100h, 0.95thr',
           '25h,  0.99thr', '50h,  0.99thr', '100h, 0.99thr']
size = (24, 20)
fontsize=24
linewidth=4

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

    # print(precision)
    # print(recall)

    p = len(numpy.nonzero(labels)[0])  # number of positives
    n = len(labels) - p  # number of negatives
    fp = fpr * p  # false positves
    tp = tpr * p  # true positives
    tn = n - fp  # true negatives
    fn = p - tp  # false negatives

    specificity = tn / n
    sensitivity = tpr  # same as recall

    precision2 = tp / (tp + fp)
    print(p, n)

    print('num positives: {}, num negatives: {}'.format(p, n))
    print('ROC score: {:.2f}'.format(roc_score))

    legends[i] += " AUC:{:.2f}".format(roc_score)

    filename = sys.argv[0]
    filename = os.path.splitext(filename)[0]
    filename = "/home/giulio/"

    color = cm.Spectral(i/len(files),1)


    # ROC plot
    plt.figure(1, figsize=size)
    plt.plot(fpr, tpr, color=color, linewidth=linewidth)
    plt.legend(legends, shadow=True, fancybox=True, loc=1, fontsize=fontsize)
    plt.xlabel('false positives rate', fontsize=fontsize)
    plt.ylabel('true positives rate', fontsize=fontsize)
    plt.setp(plt.gca().get_xticklabels(), rotation='horizontal', fontsize=fontsize)
    plt.setp(plt.gca().get_yticklabels(), rotation='horizontal', fontsize=fontsize)

    save_multiple_formats(filename+'roc')

    # RECALL-PRECISION plot
    plt.figure(2, figsize=size)
    plt.plot(recall, precision, color=color, linewidth=linewidth)
    # plt.plot(sensitivity, precision2, colors[i], linewidth=linewidth)
    plt.legend(legends, shadow=True, fancybox=True, loc=1, fontsize=fontsize)
    plt.xlabel('recall', fontsize=fontsize)
    plt.ylabel('precision', fontsize=fontsize)
    plt.setp(plt.gca().get_xticklabels(), rotation='horizontal', fontsize=fontsize)
    plt.setp(plt.gca().get_yticklabels(), rotation='horizontal', fontsize=fontsize)

    save_multiple_formats(filename+'recall_precision')


    # SPECIFICITY-SENSITIVITY plot
    plt.figure(3, figsize=size)
    plt.plot(specificity, sensitivity, color=color, linewidth=linewidth)
    plt.legend(legends, shadow=True, fancybox=True, loc=1, fontsize=fontsize)
    plt.ylabel('sensitivity', fontsize=fontsize)
    plt.xlabel('specificity',fontsize=fontsize)
    plt.setp(plt.gca().get_xticklabels(), rotation='horizontal', fontsize=fontsize)
    plt.setp(plt.gca().get_yticklabels(), rotation='horizontal', fontsize=fontsize)


    save_multiple_formats(filename+'specificity_sensitivity')

    i += 1

plt.show()
