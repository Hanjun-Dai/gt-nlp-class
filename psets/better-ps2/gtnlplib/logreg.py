from scipy.misc import logsumexp #hint
''' your code '''
import operator
from  gtnlplib.constants import ALL_LABELS
from collections import defaultdict, Counter
from gtnlplib.clf_base import evalClassifier, argmax
import numpy as np  
import gtnlplib.scorer as scorer

# compute the normalized probability of each label 
def computeLabelProbs(instance,weights,labels):
    probs = defaultdict(float)    
    scores = np.zeros(len(labels))
    
    for i in range(len(labels)):
        label = labels[i]
        scores[i] = 0.0
        for w in instance:
            scores[i] += instance[w] * weights[(label, w)]         
        
    logsum = logsumexp(scores)
    for i in range(len(labels)):
        probs[labels[i]] = np.exp(scores[i] - logsum)
    return probs

def trainLRbySGD(N_its,inst_generator, outfile, devkey, learning_rate=1e-4, regularizer=1e-2):
    weights = defaultdict(float)
    dv_acc = [None]*N_its
    tr_acc = [None]*N_its

    # this block is all to take care of regularization
    ratereg = learning_rate * regularizer
    def regularize(base_feats,t):
        for base_feat in base_feats:
            for label in ALL_LABELS:
                weights[(label,base_feat)] *= (1 - ratereg) ** (t-last_update[base_feat])
            last_update[base_feat] = t

    for it in xrange(N_its):
        tr_err = 0
        last_update = defaultdict(int) # reset, since we regularize at the end of every iteration
        for i,(inst,true_label) in enumerate(inst_generator):
            # apply "just-in-time" regularization to the weights for features in this instance
            regularize(inst,i)
            # compute likelihood gradient from this instance
            probs = computeLabelProbs(inst,weights,ALL_LABELS)                               
            # your code for updating the weights goes here
            if true_label != argmax(probs): tr_err += 1
            for w in inst:
                for pred in ALL_LABELS:
                    weights[(pred, w)] -= learning_rate * probs[pred] * inst[w]
                    if pred == true_label:
                        weights[(pred, w)] += learning_rate * inst[w] 
                        
        # regularize all features at the end of each iteration
        regularize([base_feature for label,base_feature in weights.keys()],i)
        
        dv_acc[it] = scorer.accuracy(evalClassifier(weights, outfile, devkey))
        tr_acc[it] = 1. - tr_err/float(i)
        print it,'dev:',dv_acc[it],'train:',tr_acc[it]
    return weights,tr_acc,dv_acc
