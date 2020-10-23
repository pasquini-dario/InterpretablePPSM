BATCH_SIZE = 1024
MAX_LEN = 16
CHARMAP = './charmap.pickle'
TERMINAL_SYMBOL = False
ENCODING = 'ascii'
#---------------------------

from inference import Inference
import myPickle, sys
import tensorflow as tf

def read_passwords(path, encoding=ENCODING, MIN_LEN=0, MAX_LEN=16):
    X = []
    with open(path, encoding=encoding, errors='ignore') as f:
        for x in f:
            x = x[:-1]
            if len(x) <= MAX_LEN and len(x) >= MIN_LEN:
                X.append(x)
    return X

def write_tsv(output, X, P, encoding=ENCODING):
    assert len(X) == len(P)
    n = len(X)
    with open(output, 'w', encoding=encoding) as f:
        for x, p in zip(X, P):
            print("%s\t%f" % (x, p), file=f)
            
            
            
if __name__ == '__main__':
    try:
        model_path = sys.argv[1]
        password_file = sys.argv[2]
        output_path = sys.argv[3]
    except:
        print("USAGE: model_path.h5 password_path.txt output_path.txt")
        sys.exit(1)
    
    
    X = read_passwords(password_file)
    cm = myPickle.load(CHARMAP)

    model = tf.keras.models.load_model(model_path, compile=False)
    S = Inference(model, cm, MAX_LEN, BATCH_SIZE)
    
    logP = S.applyBatch(X, TERMINAL_SYMBOL)
    write_tsv(output_path, X, logP)