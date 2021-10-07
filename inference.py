import tensorflow as tf
import numpy as np
from input_pipeline import MASK, string2idx, idx2string
import tqdm


class Inference:
    
    def __init__(self, model, cm, max_len, batch_size):
        self.model = model
        self.max_len = max_len
        self.batch_size = batch_size
        
        self.cm = cm
        self.cm_ = np.array( [x[0] for x in sorted(self.cm.items(), key=lambda x: x[1])] )
        

    def makeHoles(self, x, INCLUDE_END_SYMBOL):
        xl = len(x)
        oxi = string2idx(x, self.cm, self.max_len, 0, INCLUDE_END_SYMBOL)
        xi = np.tile(oxi, (xl, 1))
        for i in range(xl):
            xi[i, i] = MASK
        return oxi, xi
    
    def un_p(self, xi, p, scalar=False):
        mp = np.zeros(len(p))
        for i in range(len(p)):
            t = xi[i]
            mp[i] = p[i][i][t]
            
        if scalar:
            return np.log(mp).sum()
        
        return mp
    
    def __call__(self, s, INCLUDE_END_SYMBOL):
        
        oxi, xi = self.makeHoles(s, INCLUDE_END_SYMBOL)
        
        out = self.model(xi, training=False)
        p = out[1]

        un_p = self.un_p(oxi, p)
    
        return un_p
    
    
    def _alppyBatch(self, H, X, L):
        
        n = len(X)
        
        out = self.model(H, training=False)
        p_ = out[1]

        SCOREs = [None] * n
        tot = 0
        
        for i in range(n):
            l = L[i]
            pi = p_[tot:tot+l]
            assert len(pi) == l
            SCOREs[i] = self.un_p(X[i], pi, scalar=True)
            tot += l
        return SCOREs
    
    def applyBatch(self, _X, INCLUDE_END_SYMBOL):
        H = []
        X = []
        L = []
        SCORE = []

        for i, x_ in tqdm.tqdm( list( enumerate(_X) ) ):
            oxi, xi = self.makeHoles(x_, INCLUDE_END_SYMBOL)
            L.append(len(x_))
            X.append(oxi)
            H.append(xi)

            if i and i % self.batch_size == 0:
                H = np.concatenate(H)
                SCORE += self._alppyBatch(H, X, L)
                H = []
                X = []
                L = []
        if H:
            H = np.concatenate(H)
            SCORE += self._alppyBatch(H, X, L)

        return SCORE
    
    def sorting(self, X):
        up = np.array(self.applyBatch(X))
        perm = np.argsort(-up)
        return perm