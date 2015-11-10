import numpy as np #hint: np.log
from itertools import chain
import operator
from collections import defaultdict, Counter
from gtnlplib.preproc import conllSeqGenerator

from gtnlplib import scorer
from gtnlplib import constants
from gtnlplib import preproc
from gtnlplib.constants import START_TAG ,TRANS ,END_TAG , EMIT

argmax = lambda x : max(x.iteritems(),key=operator.itemgetter(1))[0]

# define viterbiTagger
start_tag = constants.START_TAG
trans = constants.TRANS
end_tag = constants.END_TAG
emit = constants.EMIT

def ensembleViterbi(words, one_feat_func, one_weights, two_feat_func, two_weights, coeff, all_tags, debug=False):
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
                one_feat = one_feat_func(words, tag, prev_tag, idx)
                two_feat = two_feat_func(words, tag, prev_tag, idx)
                
                cur_score = coeff * get_score(one_feat, one_weights) + (1 - coeff) * get_score(two_feat, two_weights) + prev_stats[prev_tag]
                if (not tag in trellis[idx]) or (cur_score > trellis[idx][tag]):
                    trellis[idx][tag] = cur_score
                    pointers[idx][tag] = prev_tag    

    last_tag = None
    for tag in all_tags:
        one_feat = one_feat_func(words, END_TAG, tag, len(words))
        two_feat = two_feat_func(words, END_TAG, tag, len(words))
                
        total_score = coeff * get_score(one_feat, one_weights) + (1 - coeff) * get_score(two_feat, two_weights) + trellis[len(words) - 1][tag]
        if total_score > best_score:
            best_score = total_score
            last_tag = tag

    idx = len(words) - 1
    while last_tag != START_TAG:
        output[idx] = last_tag
        last_tag = pointers[idx][last_tag]
        idx -= 1           
    return output,best_score
                
def viterbiTagger(words,feat_func,weights,all_tags,debug=False):
    """
    :param words: list of words
    :param feat_func: feature function
    :param weights: defaultdict of weights
    :param tagset: list of permissible tags
    :param debug: optional debug flag
    :returns output: tag sequence
    :returns best_score: viterbi score of best tag sequence
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