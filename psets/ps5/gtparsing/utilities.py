import numpy as np
from collections import defaultdict

# Implement for deliverable 2a
def CPT (instances, htag):
    """ Accepts instances which is a list and a tag index.
        Computes the conditional probability of modifier given the head tag.

        params:
        instances: list
        htag: integer

        returns:
        output: Dict - where key is a tag and the value is probability.
    """        
    total = 0
    output = defaultdict(float)
    for instance in instances:
        for i in range(len(instance.words)):
            if instance.heads[i] >= 0 and instance.pos[instance.heads[i]] == htag:
                output[instance.pos[i]] += 1.0
                total += 1.0
    for w in output:
        output[w] /= float(total) 
    return output


def entropy (distr):
    """ Calculates the entropy of a given distribution """
    ans = 0.0
    for w in distr:
        ans -= distr[w] * np.log(distr[w])
    return ans
