import numpy as np

def logw2w(logw):
    '''
    convert log weightd to normalized weights
    
    logw: nparray vector
    '''
    logw = logw - np.max(logw)
    w = np.exp(logw)
    sumw = np.sum(w)
    w = w / sumw
    return w
    
def logmean(logw):
    '''
    return the log of mean of weights given the log weights
    
    logw: nparray vector
    return: float
    '''
    log_scale = np.max(logw)
    logw = logw - log_scale
    w = np.exp(logw)
    sumw = np.sum(w)
    w = w / sumw
    log_scale = log_scale + np.log(sumw)
    m = np.mean(w)
    return np.log(m) + log_scale

def weightedsum(x,w):
    '''
    Weighted sum of vectors
    
    x: M*D, nparray
    w: M, nparray
    
    accu: D, nparray
    '''
    accu = np.ones(np.size(x,1))*0
    for i in range(len(w)):
        accu += w[i]*x[i,:]
    return accu