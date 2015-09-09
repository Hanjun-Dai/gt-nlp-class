#!/bin/bash

tr -sc 'A-Za-z' '[\012*]' < pg2701.txt > words.case
tail -n +2 words.case > nextwords.case
paste words.case nextwords.case | sort | uniq -c > bigram.case


cat bigram.case | grep -P 'Sperm\tWhale'
cat words.case | sort | uniq -c | grep -w 'Sperm'
cat words.case | sort | uniq -c | grep -w 'Whale'
