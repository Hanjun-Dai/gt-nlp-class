import numpy as np #hint: np.log
from itertools import chain
from collections import defaultdict, Counter
from gtnlplib.preproc import dataIterator
from gtnlplib.constants import OFFSET, TRAINKEY, DEVKEY
from gtnlplib import scorer
from gtnlplib.clf_base import evalClassifier


''' keep the shell '''
def learnNBWeights(counts, class_counts, allkeys, alpha=0.1):
    weights = defaultdict(int)
    num_instances = float(sum(class_counts.values()))
    V = len(allkeys) - 1
    for label in class_counts:
        tokens_current = float(sum(counts[label].values())) - counts[label][OFFSET]
        for w in allkeys:
            prob = float(counts[label][w] + alpha) / (tokens_current + alpha * V)
            weights.update({(label, w) : np.log(prob)})
        weights.update({(label, OFFSET) : np.log(float(class_counts[label]) / num_instances) })
    return weights

def regularization_using_grid_search (alphas, counts, class_counts, allkeys, tr_outfile='nb.alpha.tr.txt', dv_outfile='nb.alpha.dv.txt'):
    tr_accs = []
    dv_accs = []
    # Choose your alphas here
    weights_nb_alphas = dict()
    for alpha in alphas:
        weights_nb_alphas[alpha] = learnNBWeights(counts, class_counts, allkeys, alpha)
        confusion = evalClassifier(weights_nb_alphas[alpha],tr_outfile,TRAINKEY)
        tr_accs.append(scorer.accuracy(confusion))
        confusion = evalClassifier(weights_nb_alphas[alpha],dv_outfile,DEVKEY)
        dv_accs.append(scorer.accuracy(confusion))
    return weights_nb_alphas, tr_accs, dv_accs
