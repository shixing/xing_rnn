# generate the 1 + 1 = 2: source 1 + 1, target 2

import sys
import os
import numpy as np

def generate(fsource, ftarget, n):
    # at most 5
    fs = open(fsource,'w')
    ft = open(ftarget,'w')
    for i in xrange(n):
        a = np.random.randint(0,200)
        b = np.random.randint(0,200)
        op = np.random.randint(0,2)
        ops = "+"
        if op == 0:
            c = a + b
            ops = "+"
        else:
            c = a + b
            ops = "+"
        a_str = str(a)
        b_str = str(b)
        c_str = str(c)
        source = a_str + ops + b_str
        source = ' '.join(source)
        target = " ".join(c_str)
        fs.write(source + "\n")
        ft.write(target + "\n")
    fs.close()
    ft.close()

def main():
    folder = sys.argv[1]

    fns = ['train','valid','test']
    ns = [1000,100,100]

    for fn,n in zip(fns,ns):
        generate(os.path.join(folder,fn+'.src'),os.path.join(folder,fn+'.tgt'),n)

if __name__ == "__main__":
    main()
