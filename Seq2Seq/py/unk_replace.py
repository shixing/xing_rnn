from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys


def load_lex(lex_path):
    # lex_path: str
    # source_vocab: dict {"word":index}
    # target_vocab: dict {"word":index}
    ttable = {}
    f = open(lex_path)
    for line in f:
        ll = line.split()
        s = ll[1]
        t = ll[0]
        p = float(ll[2])
        if s not in ttable:
            ttable[s] = (t, p)
        max_t, max_p = ttable[s]
        if p > max_p:
            ttable[s] = (t,p)
    f.close()
    return ttable

if __name__ == "__main__":
    ttable = load_lex(sys.argv[1])
    for s in ttable:
        t,p = ttable[s]
        print(s, t, p)
