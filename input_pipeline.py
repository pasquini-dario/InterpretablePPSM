import numpy as np
import tensorflow as tf
import os
import myPickle


XNAME = 'X.txt'
CMNAME = 'charmap.pickle'
ENCODING = 'ascii'
buffer_size = 10000


# Special tokens
MASK = 3
END = 2
PAD = 0
################

MAX_MASKED = .3

#def mask(x, xl):
#    if INCLUDE_END_SYMBOL: xl -= 1
#    k = np.random.randint(0, xl)
#    y = x[k]
#    x[k] = MASK
#    return x, [y], [k]

def mask(x, xl, MAX_MASKED, INCLUDE_END_SYMBOL):
    if INCLUDE_END_SYMBOL: xl -= 1
        
    if MAX_MASKED == -1:
        # single missing
        kk = [np.random.randint(0, xl)]
    else:
        num_masked = int(xl * MAX_MASKED)
        removed = []
        kk = np.random.randint(0, xl, size=num_masked)
        
    for k in kk:
        x[k] = MASK
        
    return x, [], kk

def idx2string(P, CM_):
    return ''.join([CM_[p] for p in P if p > 0 and p != END])

def string2idx(x, CM, MAX_LEN, CMm, INCLUDE_END_SYMBOL):
    f = lambda x: CM[x] if x in CM else CMm
    x = list(map(f, x))
    if INCLUDE_END_SYMBOL:
        x += [END]
    x += [0] * (MAX_LEN-len(x))
    return np.array(x)

def makeIterInput(home, batch_size, MAX_MASKED, INCLUDE_END_SYMBOL, MAX_LEN=32, buffer_size=buffer_size, for_prediction=False):
    XPATH = os.path.join(home, XNAME) 
    
    CMPATH = os.path.join(home, CMNAME)     
    CM = myPickle.load(CMPATH)
    vocab_size = max(CM.values()) + 1
        
    def G(*args):
        # for each chunk
        with open(XPATH, encoding=ENCODING, errors='ignore') as f:
            for x in f:
                
                x = x[:-1]
                xl = len(x)
                
                #if not INCLUDE_END_SYMBOL: print("NO <END>")
                
                if xl > MAX_LEN - int(INCLUDE_END_SYMBOL):
                    continue
                    
                xi = string2idx(x, CM, MAX_LEN, vocab_size, INCLUDE_END_SYMBOL)
                
                xi_in, _, kk = mask(xi.copy(), xl, MAX_MASKED, INCLUDE_END_SYMBOL)
                prediction_mask = np.zeros(MAX_LEN, np.int32)

                for k in kk:
                    prediction_mask[k] = 1
                
                xi_out = xi
                                                        
                yield xi_in, prediction_mask, xi_out
            
    dataset = tf.data.Dataset.from_generator(G, (tf.int32, tf.int32, tf.int32) , ((None,), (None,), (None,)))
        
    if not for_prediction:
        dataset = dataset.shuffle(buffer_size)
        
    dataset = dataset.padded_batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=buffer_size)
    
    return dataset, vocab_size+1, CM


