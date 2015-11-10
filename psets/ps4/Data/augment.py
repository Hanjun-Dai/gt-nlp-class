import os
import sys
import random

if __name__ == '__main__':
    times = int(sys.argv[1])
    frac = float(sys.argv[2])

    sentences = []
    with open('bak-oct27.train', 'r') as f:
        sent = []
        for line in f:
            if len(line.strip()) == 0:
                sentences.append(sent)
                sent= []
            else:
                sent.append(line)
    
    with open('oct27.train', 'w') as f:
        for t in range(times):
            for sent in sentences:
                for l in sent:
                    if random.random() > frac:
                        continue
                    f.write(l)
                f.write('\n')
                

