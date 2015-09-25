''' your code '''
import operator
from  constants import *
from collections import defaultdict, Counter
from gtnlplib import preproc
import scorer
from gtnlplib import constants
from gtnlplib import clf_base

argmax = lambda x : max(x.iteritems(),key=operator.itemgetter(1))[0]

def get_tags(trainfile):
    counters = defaultdict(Counter)
    with open(trainfile) as instances:
        for line in instances:
            if len(line.rstrip()) == 0:
                continue
            
            parts = line.rstrip().split()
            cur_word = parts[0]
            if len(parts)>1:
                cur_tag = parts[1]
            else: 
                cur_tag = UNKNOWN
            
            counters[cur_tag][cur_word] += 1
                
    """Produce a Counter of occurences of word in each tag"""
    return counters

def get_noun_weights():
    """Produce weights dict mapping all words as noun"""
    weights = {('N', OFFSET) : 1}
    return weights

def get_most_common_weights(trainfile):
    counters = defaultdict(Counter)
    with open(trainfile) as instances:
        for line in instances:
            if len(line.rstrip()) == 0:
                continue
            
            parts = line.rstrip().split()
            cur_word = parts[0]
            if len(parts)>1:
                cur_tag = parts[1]
            else: 
                cur_tag = UNKNOWN
                
            counters[cur_word][cur_tag] += 1
    
    weights = {}
    for word, cnt in counters.iteritems():
        tag = counters[word].most_common(1)[0][0]
        weights[(tag, word)] = 1.0     
    return weights

def get_class_counts(counters):
    class_counts = defaultdict(int) 
    for tag, cnt in counters.iteritems():
        class_counts[tag] = sum(cnt.values())
    return class_counts
