import math
import tqdm
import os
import pickle
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import re

EMPTY = '\t'

getnettemps =  lambda p, t: np.where(p.argsort()[::-1]==t)

def load(filename, python2=False, **kargs):
    if python2:
        kargs.update(encoding='latin1')
    with open(filename, 'rb') as f:
        data = pickle.load(f, **kargs)
    return data

class Inference:
    
    def makeHoles(self, x):
        l = len(x)
        if self.include_last and l != self.x_len:
            l += 1
        xi = self.parseString(x)
        xi = np.tile(xi, (l, 1))
        for i in range(l):
            xi[i, i] = -1
        return xi
        
    
    def parseString(self, C):
        I = np.zeros(self.x_len, np.int32)
        for i in range(len(C)):
            c = C[i]
            if c == EMPTY:
                I[i] = -1
            else:
                if not c in self.cm:
                    I[i] = 0
                else:
                    I[i] = self.cm[c]
        return I[None, :]
    
    def toS(self, P):
        return ''.join([self.cm_[p] for p in P if p > 0]) 
            

    def __init__(self, mpath, include_last=False, batch_size=4096):
        self.mpath = mpath
        self.batch_size = batch_size
        self.include_last = include_last
        
        cm_path = os.path.join(mpath, 'char_map.pickle')
        self.cm = load(cm_path)
        self.cm_ = np.array( [x[0] for x in sorted(self.cm.items(), key=lambda x: x[1])] )
        
        tf.logging.set_verbosity(tf.logging.ERROR)
        module = hub.Module(mpath)
        out_shape = module.get_output_info_dict()['p'].get_shape().as_list()
        self.x_len = out_shape[1]
        
        self.x = tf.placeholder(tf.int32, shape=(None, self.x_len))
        # simple inference
        out = module(self.x, as_dict=True)
        self.infp = out['p']
        self.infx = out['x']
        self.inflogits = out['logits']
        self.infprediction_string = out['prediction_string']
        
        ###
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        ####
    
    def score(self, p, s):
        
        l = len(s)
        if self.include_last and l != self.x_len:
            l += 1
        
        c = np.zeros(l, np.float64)
        
        A = []
        
        for j in range(l):
            if self.include_last and j == l-1:
                t = 0
            else:
                if not s[j] in self.cm:
                    t = len(self.cm)-1
                else:
                    t = self.cm[s[j]]
            
            nattempt = getnettemps(p[j][j], t)
            A.append(nattempt)
            
            c[j] = (p[j][j][t])
                
        A = np.concatenate(A).ravel() + 1
        return c, A

    def infereRaw(self, H, sess=None):
        n = len(H)
        
        do = lambda sess, H: sess.run([self.infx, self.infp], {self.x:H})
        
        if sess:
            _, p_ = do(sess, H)
        else:
            print("Creating tf.Session")
            sess = tf.Session(config=self.config)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            _, p_ = do(sess, H)
                
        return p_, sess
    
    def _meterBatch(self, H, X, sess=None):
        n = len(X)
        
        do = lambda sess, H: sess.run([self.infx, self.infp], {self.x:H})
        
        if sess:
            _, p_ = do(sess, H)
        else:
            print("Creating tf.Session")
            with tf.Session(config=self.config) as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.tables_initializer())
                _, p_ = do(sess, H)
            
        SCOREs = [None] * n
        tot = 0
        for i in range(n):
            l = len(X[i])
            if self.include_last and l != self.x_len:
                l += 1
            pi = p_[tot:tot+l]
            assert len(pi) == l
            SCOREs[i] = self.score(pi, X[i])
            tot += l
        return SCOREs
        
        
    def meterBatched(self, _X):
        H = []
        X = []
        SCORE = []
        
        with tf.Session(config=self.config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
        
            for i, x_ in tqdm.tqdm( list( enumerate(_X) ) ):
                X.append(x_)
                xi = self.makeHoles(x_)
                H.append(xi)

                if i and i % self.batch_size == 0:
                    H = np.concatenate(H)
                    SCORE += self._meterBatch(H, X, sess)
                    H = []
                    X = []
                    
            if H:
                H = np.concatenate(H)
                SCORE += self._meterBatch(H, X, sess)

        return SCORE
        

    def meterSingle(self, s, sess=None):
        
        xi = self.makeHoles(s)
        
        if sess is None:
            sess = tf.Session(config=self.config)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
        x_, p_, prediction_string_ = sess.run([self.infx, self.infp, self.infprediction_string], {self.x:xi})
            
        c = self.score(p_, s)
            
        prediction_string_ = [''.join(ss.decode().split('\n')) for ss in prediction_string_]
        
        return prediction_string_, c, p_, sess
    
    
    def characterGuess(self, s, sess=None, include_last=False):
        if include_last:
            ss = s + '\n'
        
        xs, cp, p, sess = self.meterSingle(s, sess=sess)
            
        C = []
        for i, pp in enumerate(p):
            perm = pp[i].argsort(0)[::-1]
            ci = self.cm_[perm].tolist()
            C.append(ci)
            
        return C, cp[0], p, sess