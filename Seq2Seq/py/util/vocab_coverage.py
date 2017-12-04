# python vocab_coverage.py test.txt vocab.txt

def load_vocab(fn):
    f = open(fn)
    d = {}
    for line in f:
        line = line.strip()
        d[line] = 1
    f.close()
    return d

def coverage(fn, vocab):
    f = open(fn)
    s = 0
    n = 0
    for line in f:
        words = line.split()
        s += len(words)
        for word in words:
            if word in vocab:
                n += 1
    f.close()
    print n, s, n*1.0/s

if __name__ == "__main__":
    import sys
    vocab = load_vocab(sys.argv[2])
    coverage(sys.argv[1],vocab)
    
    
