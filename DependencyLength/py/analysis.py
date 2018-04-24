from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


# load data
def load_pickle(fn):
    f = open(fn)
    s = cPickle.load(f)
    f.close()

    # s = [     ] * 347
    # s[0] = [LSTMTuple_0, LSTMTuple_1]
    
    return s 

def print_dim(s,ct_ht,n_layer,dim):
    for i in xrange(len(s)):
        for j in xrange(s[i][n_layer].c.shape[1]):
            if ct_ht == 'ct':
                print(s[i][n_layer].c[0, j, dim])
            else:
                print(s[i][n_layer].h[0, j, dim])
        print()


# transform
def concatenate(s):
    l = 0
    n_layer = len(s[0])
    n_dim = s[0][0].c.shape[2]
    for i in xrange(len(s)):
        l += s[i][0].c.shape[1]
    print("Total length:", l)
    print('Number of layer:', n_layer)
    print('Dim:', n_dim)

    c = []
    h = []

    for i in xrange(n_layer):
        ct = np.zeros((l, n_dim))
        ht = np.zeros((l, n_dim))
        start = 0
        for j in xrange(len(s)):
            _c = s[j][i].c
            _h = s[j][i].h
            _l = _c.shape[1]
            ct[start:start+_l,:] = _c
            ht[start:start+_l,:] = _h
            #print(start, start+_l)
            start += _l
        ct = ct.T
        ht = ht.T
        c.append(ct)
        h.append(ht)
        #print(ct)
        #print(ht)
    return c, h

# fft
def transform_fft(m):
    # m = [dim, l]
    f = np.abs(np.fft.rfft(m))
    freq = np.fft.rfftfreq(m.shape[1])
    return f,freq

# visualize it; 
def visulize(f,freq, d, s, fig_name):
    #print(f[-5,:])
    #print_dim(s,'ht',0,294)

    sp = f 
    fs = np.zeros(sp.shape)
    for i in xrange(fs.shape[0]):
        fs[i] = freq
    sp += 1e-10
    av = np.average(fs, axis = -1, weights = sp)
    average_fs = np.mean(av)
    sorted_av_index = np.argsort(av)
    #print(sorted_av_index)
    #i = sorted_av_index[2]

    
    
    sorted_av = sorted(av)
    x = range(sp.shape[0])
    y = sorted_av

    plt.bar(x,y,0.001)
    plt.axhline(average_fs)
    #plt.show()
    plt.savefig(fig_name)
    plt.gcf().clear()



if __name__ == "__main__":
    folder = sys.argv[1]
    fn = os.path.join(folder, "b12.dump_lstm.pickle")
    s = load_pickle(fn)
    c, h = concatenate(s)

    for t in ["ct","ht"]:
        for l in xrange(2):
            fig_name = "{}_{}.pdf".format(t,l)
            print(fig_name)
            fig_path = os.path.join(folder,fig_name)
            data = c[l]
            if t == 'ht':
                data = h[l]
            f,freq = transform_fft(data)
            visulize(f,freq,data,s,fig_path)    
