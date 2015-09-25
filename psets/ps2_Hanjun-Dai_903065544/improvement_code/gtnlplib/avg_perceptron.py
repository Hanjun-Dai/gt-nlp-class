import operator
from  constants import *
from collections import defaultdict, Counter
from clf_base import predict, evalClassifier
import scorer

def trainAvgPerceptron(N_its,inst_generator,labels, outfile, devkey):
    tr_acc = [None]*N_its #holder for training accuracy
    dv_acc = [None]*N_its #holder for dev accuracy
    weights = defaultdict(float) 
    avg_weights = defaultdict(float)
    wsum = defaultdict(float)
    cur_T = 1
    for i in xrange(N_its):
        weights,wsum, tr_err,tr_tot = oneItAvgPerceptron(inst_generator,weights,wsum,labels,cur_T) #call your function for a single iteration
        cur_T += tr_tot
        for w in wsum:
            avg_weights[w] = weights[w] - wsum[w] / float(cur_T)    
        confusion = evalClassifier(avg_weights, outfile, devkey) #evaluate on dev data
        dv_acc[i] = scorer.accuracy(confusion) #compute accuracy
        tr_acc[i] = 1. - tr_err/float(tr_tot) #compute training accuracy from output
        print i,'dev: ',dv_acc[i],'train: ',tr_acc[i]         
    
    return avg_weights, tr_acc, dv_acc


def oneItAvgPerceptron(inst_generator,weights,wsum,labels,Tinit=0):
    errors = 0.
    num_insts = 0.
    t = Tinit
    for sample in inst_generator:
        feature, label = sample
        pred = predict(feature, weights, labels)[0]        
        num_insts += 1
        if pred != label:
            errors += 1
            for w in feature:
                weights[(label, w)] += feature[w]
                weights[(pred, w)] -= feature[w]
                wsum[(label, w)] += t * feature[w]
                wsum[(pred, w)] -= t * feature[w] 
        t += 1
    return weights, wsum, errors, num_insts
