#!/usr/bin/python
# -*- coding: utf-8 -*-
# process xml bracket F1 score;
# python3 bracket_f1 reference.txt generated.txt outputfolder

import sys
import os

from PYEVALB import scorer

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

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
    gold_path = os.path.join(folder, "gold.txt")
    test_path = os.path.join(folder, "test.txt")
    result_path = os.path.join(folder, "result.txt")
    
    convert(sys.argv[1],gold_path)
    convert(sys.argv[2],test_path)
    
    scorer.Scorer().evalb(gold_path, test_path, result_path)

def main2():
    from PYEVALB import parser

    gold = '(claim (claim-text (w 0) (w 1) (w 2) (w 3) (w 4) (w 5) (w 6) (w 7) (w 8) (w 9) (w 10) (w 11) (w 12) (w 13) (w 14) (w 15) (w 16) (w 17) (w 18) (w 19) (w 20) (w 21) (w 22) (w 23) (w 24) (w 25) (w 26) (w 27) (w 28) (w 29) (w 30) (w 31) (w 32) (w 33) (w 34) (w 35) (w 36) (w 37) (w 38) (w 39) (w 40) (w 41) (w 42) (w 43) (w 44) (w 45) (w 46) (w 47) (w 48) (w 49) (w 50) (w 51) (w 52) (w 53) (w 54) (w 55) (w 56) (w 57) (w 58) (w 59) ) )'

    test = '(claim-text (w 0) (w 1) (w 2) (w 3) (w 4) (w 5) (w 6) (w 7) (w 8) (w 9) (w 10) (w 11) (w 12) (w 13) (w 14) (w 15) (w 16) (w 17) (w 18) (w 19) (w 20) (w 21) (w 22) (w 23) (w 24) (w 25) (w 26) (w 27) (w 28) (w 29) (w 30) (w 31) (w 32) (w 33) (w 34) (w 35) (w 36) (w 37) (w 38) (w 39) (w 40) (w 41) (w 42) (w 43) (w 44) (w 45) (w 46) (w 47) (w 48) (w 49) (w 50) (w 51) (w 52) (w 53) (w 54) (w 55) (w 56) (w 57) (w 58) (w 59) ) )'

    gold_tree = parser.create_from_bracket_string(gold)
    test_tree = parser.create_from_bracket_string(test)

    result = scorer.Scorer().score_trees(gold_tree, test_tree)

    print result

    
if __name__ == "__main__":
    main()
