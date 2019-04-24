'''
funcitons that measure the sample qualities, sometimes with respect to a target distribution

input:
    x: N*D nparray: N sample of D dimensions
    w: N nparray: normalized weights
    pi: function handle: target distribution
        input:
            x: N*D nparray
        output:
            pix: N nparray, pi(x) values i.e. likelihood

output:
    out: float: the measured quality
'''

import numpy as np

def ESS(w):
    # effective sample size
    return 1/np.sum(w**2)

def ESSoverN(w):
    # effective sample size over N
    # return value will between 0 and 1
    N = len(w)
    return ESS(w)/N

def Dchi2(x, pi, w=None):
    if not w:
        N = np.size(x,0)
        w = np.ones(N)/N
    return np.sum(w**2/pi(x))