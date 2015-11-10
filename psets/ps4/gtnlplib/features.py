import gtnlplib.constants
from collections import defaultdict, Counter
from gtnlplib.constants import END_TAG, START_TAG, TRANS, CURR_SUFFIX, PREV_SUFFIX

common_suffix = ['able', 'ible', 'al', 'ial', 'ed', 'en', 'er', 'er,', 'est', 'ful', 'ic', 'ing', 'ion', 'tion', 'ation', 'ition', 'ity', 'ty', 'ive', 'ative', 'itive', 'less', 'ly', 'ment', 'ness', 'ous', 'eous', 'ious', 's', 'es', 'y']
common_prefix = ['anti', 'de', 'dis', 'en', 'em', 'fore', 'in', 'im', 'il', 'ir', 'inter', 'mid', 'mis', 'non', 'over', 'pre', 're', 'semi', 'sub', 'super', 'trans', 'un', 'under']

def wordFeatures(words,tag,prev_tag,m):
    '''
    :param words: a list of words
    :type words: list
    :param tag: a tag
    :type tag: string
    :type prev_tag: string
    :type m: int
    '''
    out = {(gtnlplib.constants.OFFSET,tag):1}
    if m < len(words): #we can have m = M, for the transition to the end state
        out[(gtnlplib.constants.EMIT,tag,words[m])]=1
    return out

def wordCharFeatures(words,tag,prev_tag,m):
    output = wordFeatures(words,tag,prev_tag,m) #start with the features from wordFeatures
    # add your code here
    if m < len(words):        
        output[(CURR_SUFFIX, tag, words[m][-1])] = 1
    if m > 0:
        output[(PREV_SUFFIX, tag, words[m-1][-1])] = 1
    return output
    
def has_nonalpha(st):
    for c in st:
        if not c.isalpha():
            return True
    return False

def yourFeatures(words,tag,prev_tag,m):
    output = wordFeatures(words,tag,prev_tag,m) #start with the features from wordFeatures
    
    if m < len(words):        
        for suffix in common_suffix: # good
            if words[m].endswith(suffix):
                output[('--CUR_SUFFIX_PATTERN--', tag, suffix)] = 1
        for prefix in common_prefix: # good
            if words[m].startswith(prefix):
                output[('--CUR_PREFIX_PATTERN--', tag, prefix)] = 1

        for i in range(1, 5):
            if i > len(words[m]):
                break
            output[('--CURR_SUFFIX_1to3--', tag, words[m][-i:])] = 1 # good
            output[('--CURR_PREFIX_1to3--', tag, words[m][0 : i])] = 1 # good             
                
        if len(words[m]) > 1:
            output[('--CURR_SEC_SUFFIX--', tag, words[m][-2])] = 1 # not sure
        
        if has_nonalpha(words[m]):
            output[('--CURR_ISALPHA--', tag, 'F')] = 1
        else:
            output[('--CURR_ISALPHA--', tag, 'T')] = 1
               
        if m < len(words) - 1:         
            output[('--NEXT_SUFFIX--', tag, words[m + 1][-1])] = 1            
            if words[m + 1].isalpha():
                output[('--NEXT_ISALPHA--', tag, 'T')] = 1
            else:
                output[('--NEXT_ISALPHA--', tag, 'F')] = 1
    if m > 0:                
        output[('--PREV_PREFIX--', tag, words[m-1][0])] = 1        
        output[('--PREV_WORD--', tag, words[m-1])] = 1 # good                          
    # your stuff
    return output

def seqFeatures(words,tags,featfunc):
    '''
    :param words: a list of words
    :param tags: a list of tags
    :param featfunc: a function to compute f(words,tag_m,tag_{m-1},m)
    :returns list of features
    '''
    allfeats = defaultdict(float)
    prev_tag = START_TAG
    for m in range(len(words)):
        feat = featfunc(words, tags[m], prev_tag, m)
        prev_tag = tags[m]
        for key in feat:
            if not key in allfeats:
                allfeats[key] = 0.0
            allfeats[key] += feat[key]
    feat = featfunc(words, END_TAG, prev_tag, len(words))
    for key in feat:
        if not key in allfeats:
            allfeats[key] = 0.0
        allfeats[key] += feat[key]
    # your code here
    return allfeats


def wordTransFeatures(words,tag,prev_tag,m):
    output = wordFeatures(words,tag,prev_tag,m) #start with the features from wordFeatures    
    output[(TRANS, tag, prev_tag)] = 1.0
    # your code here
    return output

def yourHMMFeatures(words,tag,prev_tag,m):
    output = wordTransFeatures(words,tag,prev_tag,m) #start with the features from wordFeatures
    #add smart stuff
    
    if m < len(words):
        for suffix in common_suffix: # good
            if words[m].endswith(suffix):
                output[('--CUR_SUFFIX_PATTERN--', tag, suffix)] = 1
        for prefix in common_prefix: # good
            if words[m].startswith(prefix):
                output[('--CUR_PREFIX_PATTERN--', tag, prefix)] = 1

        for i in range(1, 5):
            if i > len(words[m]):
                break
            output[('--CURR_SUFFIX_1to3--', tag, words[m][-i:])] = 1 # good
            output[('--CURR_PREFIX_1to3--', tag, words[m][0 : i])] = 1 # good 
                    
        if len(words[m]) > 1:
            output[('--CURR_SEC_SUFFIX--', tag, words[m][-2])] = 1 # not sure
        
        if has_nonalpha(words[m]):
            output[('--CURR_ISALPHA--', tag, 'F')] = 1
        else:
            output[('--CURR_ISALPHA--', tag, 'T')] = 1
               
        if m < len(words) - 1:         
            output[('--NEXT_SUFFIX--', tag, words[m + 1][-1])] = 1            
            if words[m + 1].isalpha():
                output[('--NEXT_ISALPHA--', tag, 'T')] = 1
            else:
                output[('--NEXT_ISALPHA--', tag, 'F')] = 1
    if m > 0:                
        output[('--PREV_PREFIX--', tag, words[m-1][0])] = 1        
        output[('--PREV_WORD--', tag, words[m-1])] = 1 # good    
    
    '''if m < len(words):        
        output[(CURR_SUFFIX, tag, words[m][-1])] = 1
        output[('--CURR_PREFIX--', tag, words[m][0])] = 1
        if len(words[m]) > 1:
            output[('--CURR_SEC_PREFIX--', tag, words[m][1])] = 1
        if words[m].isalpha():
            output[('--CURR_ISALPHA--', tag, 'T')] = 1
        else:
            output[('--CURR_ISALPHA--', tag, 'F')] = 1   
        if m < len(words) - 1:
            output[('--NEXT_SUFFIX--', tag, words[m + 1][-1])] = 1
            if words[m + 1].isalpha():
                output[('--NEXT_ISALPHA--', tag, 'T')] = 1
            else:
                output[('--NEXT_ISALPHA--', tag, 'F')] = 1
    if m > 0:
        output[(PREV_SUFFIX, tag, words[m-1][-1])] = 1
        output[('--PREV_PREFIX--', tag, words[m-1][0])] = 1'''
    return output