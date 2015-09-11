def get_token_type_ratio (vocabulary):
    # YOUR CODE HERE
    return float(sum(vocabulary.values())) / len(vocabulary.keys())

def type_frequency (vocabulary, k):
    # YOUR CODE HERE
    cnt = 0
    for value in vocabulary.values():
        if value == k:
            cnt += 1
    return cnt

def unseen_types (first_vocab, second_vocab):
    # YOUR CODE HERE
    set_first = set(first_vocab)
    set_second = set(second_vocab)
    return len(set_second.difference(set_first))
