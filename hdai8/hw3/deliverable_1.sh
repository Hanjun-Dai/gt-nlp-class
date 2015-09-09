#!/bin/bash

tr -sc 'A-Za-z' '[\012*]' < pg2701.txt > words.case
tail -n +2 words.case > nextwords.case
paste words.case nextwords.case | sort | uniq -c > bigram.case

wc words.case

cat bigram.case | grep -P 'Captain\tAhab'
cat words.case | sort | uniq -c | grep -w 'Captain'
cat words.case | sort | uniq -c | grep -w 'Ahab'
