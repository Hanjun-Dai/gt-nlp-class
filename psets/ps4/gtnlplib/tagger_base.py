import operator
from gtnlplib import scorer
from gtnlplib import preproc
from gtnlplib import clf_base
from gtnlplib.constants import DEV_FILE, OFFSET, TRAIN_FILE

from gtnlplib.constants import START_TAG, END_TAG

argmax = lambda x : max(x.iteritems(),key=operator.itemgetter(1))[0]
def classifierTagger(words,featfunc,weights,all_tags):
    """
    :param words: list of words
    :param features: function from lists of words and tags to list of features
    :param weights: defaultdict of weights
    :param all_tags: list of permissible tags
    :returns list of tags
    """
    def inner_product(weights, features):
        ans = 0.0
        for key in features:
            if key in weights:
                ans += features[key] * weights[key]
        return ans
       
    ''' viterbi tagger 
    out = []
    opt = []
    prev_link = []
    cur_stats = {}
    cur_links = {}    
    for tag in all_tags:
        feat = featfunc(words, tag, START_TAG, 0)
        cur_stats[tag] = inner_product(weights, feat)
        cur_links[tag] = START_TAG
    prev_link.append(cur_links)
    opt.append(cur_stats)
            
    for m in range(1, len(words)):
        cur_stats = {}
        cur_links = {}
        for cur_tag in all_tags:
            cur_stats[cur_tag] = {}            
            for prev_tag in all_tags:
                feat = featfunc(words, cur_tag, prev_tag, m)
                cur_stats[cur_tag][prev_tag] = opt[m - 1][prev_tag] + inner_product(weights, feat)
            cur_links[cur_tag] = argmax(cur_stats[cur_tag])
            cur_stats[cur_tag] = cur_stats[cur_tag][cur_links[cur_tag]]     
        opt.append(cur_stats)
        prev_link.append(cur_links)
    
    last_stats = {}
    for prev_tag in all_tags:
        feat = featfunc(words, END_TAG, prev_tag, len(words))
        last_stats[prev_tag] = inner_product(weights, feat) + opt[len(words) - 1][prev_tag]
                        
    last_tag = argmax(last_stats)
    out.append(last_tag)
    for m in reversed(xrange(len(words))):
        if m == 0:
            break
        prev_tag = prev_link[m][last_tag]
        out.append(prev_tag)
        last_tag = prev_tag
    return out[::-1]
    '''
    
    out = []
    for m in range(len(words)):
        cur_stats = {}
        for tag in all_tags:
            feat = featfunc(words, tag, 'X', m)
            cur_stats[tag] = inner_product(weights, feat)
        out.append(argmax(cur_stats))
    return out

def evalTagger(tagger,outfilename,testfile=DEV_FILE):
    """Calculate confusion_matrix for a given tagger

    Parameters:
    tagger -- Function mapping (words, possible_tags) to an optimal
              sequence of tags for the words
    outfilename -- Filename to write tagger predictions to
    testfile -- (optional) Filename containing true labels

    Returns:
    confusion_matrix -- dict of occurences of (true_label, pred_label)
    """
    alltags = set()
    for i,(words, tags) in enumerate(preproc.conllSeqGenerator(TRAIN_FILE)):
        for tag in tags:
            alltags.add(tag)
    with open(outfilename,'w') as outfile:
        for words,_ in preproc.conllSeqGenerator(testfile):
            pred_tags = tagger(words,alltags)
            for tag in pred_tags:
                print >>outfile, tag
            print >>outfile, ""
    return scorer.getConfusion(testfile,outfilename) #run the scorer on the prediction file
