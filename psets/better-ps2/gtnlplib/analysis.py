import operator

def getTopFeats(weights,class1,class2,allkeys,K=5):
    # your code here
    diff_dict = {}
    for w in allkeys:
        p = weights[(class1, w)]
        q = weights[(class2, w)]
        diff_dict[w] = p - q
    sorted_words = sorted(diff_dict.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_words[0 : K]
