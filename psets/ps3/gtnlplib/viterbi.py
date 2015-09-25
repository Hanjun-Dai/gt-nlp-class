import numpy as np #hint: np.log
from itertools import chain
import operator
from collections import defaultdict, Counter
from gtnlplib.preproc import conllSeqGenerator

from gtnlplib import scorer
from gtnlplib import most_common
from gtnlplib import preproc
from gtnlplib.constants import START_TAG, TRANS, END_TAG, EMIT
from gtnlplib import naivebayes

def argmax(scores):
    """Find the key that has the highest value in the scores dict"""
    return max(scores.iteritems(),key=operator.itemgetter(1))[0]

# define viterbiTagger
def viterbiTagger(words,feat_func,weights,all_tags,debug=False):
    """Tag the given words using the viterbi algorithm
        Parameters:
        words -- A list of tokens to tag
        feat_func -- A function of (words, curr_tag, prev_tag, curr_index)
        that produces features
        weights -- A defaultdict that maps features to numeric score. Should
        not key error for indexing into keys that do not exist.
        all_tags -- A set of all possible tags
        debug -- (optional) If True, print the trellis at each layer
        Returns:
        tuple of (tags, best_score), where
        tags -- The highest scoring sequence of tags (list of tags s.t. tags[i]
        is the tag of words[i])
        best_score -- The highest score of any sequence of tags
        """
        
    trellis = [None] * len(words)
    pointers = [None] * len(words)
    output = [None] * len(words)
    best_score = -np.inf    
    
    get_score = lambda feat, weights: sum([weights[w] for w in feat])
    
    for idx in range(len(words)):
        trellis[idx] = defaultdict(float)
        pointers[idx] = defaultdict(str)
        if idx == 0:
            prev_stats = {START_TAG : 0.0}
        else:
            prev_stats = trellis[idx - 1]         
        for tag in all_tags:
            for prev_tag in prev_stats:
                feat = feat_func(words, tag, prev_tag, idx)
                cur_score = get_score(feat, weights) + prev_stats[prev_tag]
                if (not tag in trellis[idx]) or (cur_score > trellis[idx][tag]):
                    trellis[idx][tag] = cur_score
                    pointers[idx][tag] = prev_tag
    
    last_tag = None
    for tag in all_tags:
        feat = feat_func(words, END_TAG, tag, len(words))
        total_score = get_score(feat, weights) + trellis[len(words) - 1][tag]
        if total_score > best_score:
            best_score = total_score
            last_tag = tag

    idx = len(words) - 1
    while last_tag != START_TAG:
        output[idx] = last_tag
        last_tag = pointers[idx][last_tag]
        idx -= 1           
    return output,best_score

def get_HMM_weights(trainfile):
    """Train a set of of log-prob weights using HMM transition model
        Parameters:
        trainfile -- The name of the file to train weights
        Returns:
        weights -- Weights dict with log-prob of transition and emit features
        """
    # compute naive bayes weights
    
    # convert nb weights to hmm weights
    counters = most_common.get_tags(trainfile)
    allwords = set()
    for counts in counters.values():
        allwords.update(set(counts.keys()))        
    class_counts = most_common.get_class_counts(counters)
    nb_weights = naivebayes.learnNBWeights(counters,class_counts,allwords,0.001)
    
    trans_cnt = defaultdict(Counter)
    with open(trainfile) as instances:
        prev_tag = START_TAG
        for line in instances:
            if len(line.rstrip()) == 0:
                trans_cnt[prev_tag][END_TAG] += 1
                prev_tag = START_TAG
                continue
            
            parts = line.rstrip().split()
            if len(parts) >1:
                cur_tag = parts[1]
            else: 
                cur_tag = UNKNOWN
            
            trans_cnt[prev_tag][cur_tag] += 1
            prev_tag = cur_tag                                
        if prev_tag != START_TAG:
            trans_cnt[prev_tag][END_TAG] += 1
                   
    hmm_weights = defaultdict(lambda : -1000.)
    for key in nb_weights:
        tag = key[0]
        word = key[1]
        hmm_weights[(tag, word, EMIT)] = nb_weights[key]
        
    for prev_tag in trans_cnt:
        cnt = trans_cnt[prev_tag]
        total_pairs = sum(cnt.values())        
        for cur_tag in cnt:
            hmm_weights[(cur_tag, prev_tag, TRANS)] = np.log(cnt[cur_tag]) - np.log(total_pairs) 
    
    return hmm_weights

def hmm_feats(words,curr_tag,prev_tag,i):
    """Feature function for HMM that returns emit and transition features"""
    if i < len(words):
        return [(curr_tag,words[i],EMIT),(curr_tag,prev_tag,TRANS)]
    else:
        return [(curr_tag,prev_tag,TRANS)]