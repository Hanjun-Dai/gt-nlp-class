import numpy as np #hint: np.log
from collections import defaultdict,Counter
from gtnlplib import scorer, most_common,preproc
from gtnlplib.constants import OFFSET

# this is from pset 1
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
