#!/bin/bash

tr 'a-z' 'A-Z' < pg2701.txt > pg2701_capital.txt
tr -sc 'A-Za-z' '[\012*]' < pg2701_capital.txt > words.capital
tail -n +2 words.capital > nextwords.capital
paste words.capital nextwords.capital | sort | uniq -c > bigram.capital

wc words.capital

cat bigram.capital | grep -P 'CAPTAIN\tAHAB'
cat words.capital | sort | uniq -c | grep -w 'CAPTAIN'
cat words.capital | sort | uniq -c | grep -w 'AHAB'
