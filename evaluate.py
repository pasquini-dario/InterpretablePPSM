import numpy as np

def readPC(path, MAX_LEN, encoding='ascii'):
    with open(path, encoding=encoding, errors='ignore') as f:
        raw = [x.lstrip()[:-1].split(' ') for x in f]
        raw = [[int(x[0]), ' '.join(x[1:])] for x in raw ]
        raw = [x for x in raw if x[1] and len(x[1]) <= MAX_LEN]
        F = np.array( [x[0] for x in raw] )
        X = [x[1] for x in raw]
    return X, F

def rank(f):
    f_ = {x:i for i, x in enumerate(sorted( set(f), reverse=True))}
    rank = [f_[x] for x in f]
    return rank

#++==++++===+++======
def pearsonW(a, b, w):
    wmean = lambda a, w: np.sum(w * a) / w.sum()
    var = lambda a, wma, w: np.sum(w*((a - wma)**2))

    wma = wmean(a, w)
    wmb = wmean(b, w)

    upper = np.sum( w * (a - wma) * (b - wmb) )
    bottom = np.sqrt( var(a, wma, w) * var(b, wmb, w) )
    pearson = upper / bottom
    return pearson
#++==++++===+++======
def scorePasswordsSpearmanW(pmodel, rank_gt, k):
    rank_gt = rank_gt[:k]
    w = 1 / (rank_gt + 1)
    w = w / w.sum()
    rank_model = pmodel[:k].argsort()
    score = pearsonW(rank_model, rank_gt, w)
    return score
#++==++++===+++======
def getScoreMulti(P, R, SETP_SCORE=15):
    P = -P
    S0, S1 = [], []
    for k in np.linspace(100, len(P), num=SETP_SCORE):
        k = int(k)
        rank_gt = np.arange(0, k)
        S0.append( scorePasswordsSpearmanW(P, rank_gt, k) )
        S1.append( scorePasswordsSpearmanW(P, R, k) )        
    return S0, S1
#++==++++===+++======
def getScore(P, R):
    P = -P
    S = []
    k = len(P)
    #rank_gt = np.arange(0, k)
    #S0.append( scorePasswordsSpearmanW(P, rank_gt, k) )
    S.append( scorePasswordsSpearmanW(P, R, k) )        
    return S


