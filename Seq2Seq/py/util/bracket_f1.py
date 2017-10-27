#!/usr/bin/python
# -*- coding: utf-8 -*-
# process xml bracket F1 score;

import sys
import os
from evalb import score_corpus


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def convert_sentence(line):
    line = line.decode("utf8")
    words = line.strip().split()
    new_words = []        
    i = 0
    for w in words:
        if w == "<claim>":
            w = "(claim"
        elif w == "</claim>":
            w = ")"
        elif w == "<claim-text>":
            w = "(claim-text"
        elif w == "</claim-text>":
            w = ")"
        else:
            w = "(w {})".format(i)
            i += 1
        new_words.append(w)

    line = u" ".join(new_words)
    return line, i

        
def convert_pair(fngold,fntest,fngoldout,fntestout, max_length = -1, min_length = -1):
    print(max_length, min_length)
    fgold = open(fngold)
    ftest = open(fntest)
    fgo = open(fngoldout,'w')
    fto = open(fntestout,'w')

    while True:
        line_gold = fgold.readline()
        line_test = ftest.readline()
        if not line_gold:
            break
        line_gold, lg = convert_sentence(line_gold)
        line_test, lt = convert_sentence(line_test)

        if max_length >= 0 and min_length >= 0 and lg <= max_length and lg > min_length:
            fgo.write(line_gold.encode('utf8')+"\n")
            fto.write(line_test.encode('utf8')+"\n")
        
    fgold.close()
    ftest.close()
    fgo.close()
    fto.close()

        
def convert(fnin,fnout):
    fin = open(fnin)
    fout = open(fnout,'w')

    for line in fin:
        line = line.decode("utf8")
        words = line.strip().split()
        new_words = []        
        i = 0
        for w in words:
            if w == "<claim>":
                w = "(claim"
            elif w == "</claim>":
                w = ")"
            elif w == "<claim-text>":
                w = "(claim-text"
            elif w == "</claim-text>":
                w = ")"
            else:
                w = "(w {})".format(i)
                i += 1
            new_words.append(w)

        line = u" ".join(new_words)
        fout.write(line.encode('utf8')+"\n")
        
    fout.close()
    fin.close()


def main():
    folder = sys.argv[3]
    mkdir(folder)


    # total; 
    gold_path = os.path.join(folder, "gold.txt")
    test_path = os.path.join(folder, "test.txt")
    result_path = os.path.join(folder, "result.txt")
    
    convert(sys.argv[1],gold_path)
    convert(sys.argv[2],test_path)
    
    score_corpus(gold_path, test_path, result_path)

    # buckets those sentences by length;
    buckets = [0, 20,40,80,120,160,200,400,1000]
    for i in xrange(1, len(buckets)):
        l = buckets[i]
        gold_path = os.path.join(folder, "gold{}.txt".format(l))
        test_path = os.path.join(folder, "test{}.txt".format(l))
        result_path = os.path.join(folder, "result{}.txt".format(l))
        
        convert_pair(sys.argv[1],sys.argv[2],gold_path, test_path, max_length = buckets[i], min_length = buckets[i-1])
        score_corpus(gold_path, test_path, result_path)
    

    
if __name__ == "__main__":
    main()
