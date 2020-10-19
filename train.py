MODEL_OUT = 'HOME/MODELs'
LOG_OUT = 'HOME/LOGs'


import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU') 

import  os, sys, gin

import modelCNN as model
import input_pipeline
from trainer import Trainer
import architectureCNN as architecture


def basenameNoExt(path, sep='.'):
    name = os.path.basename(path)
    return name.split(sep)[0]


@gin.configurable
def setup(name, MODEL_TYPE, home_train, home_tests, max_epochs, log_freq, MAX_LEN, hparams):
    check_train_dir = os.path.join(LOG_OUT, name)
    check_test_dir = os.path.join(check_train_dir, 'eval')
    
    MAX_MASK =  hparams['masked_chars']
    INCLUDE_END_SYMBOL = hparams['append_end']
    
    train_batch, vocab_size, cm = input_pipeline.makeIterInput(home_train, hparams['batch_size'], MAX_MASK, INCLUDE_END_SYMBOL, MAX_LEN)
    
    optimizer = tf.keras.optimizers.Adam(hparams['learning_rate'])
        
    f, train_step, predict_step = model.make_train_predict(hparams, optimizer, vocab_size, MAX_LEN)

    model_mem_footprint = (f.count_params() * 4) // (10 ** 6)
    print("\t MODEL_MEM: %dMB" % model_mem_footprint)

    t = Trainer(
            f,
            MAX_LEN,
            cm,
            train_step,
            predict_step,
            max_epochs,
            train_batch,
            home_tests,
            optimizer,
            check_train_dir,
            check_test_dir,
            1,
            log_freq,
            hparams,
    )

    
    print("TRAIN")
    t()
    print("EXPORT")
    mpath = os.path.join(MODEL_OUT, name+'.h5')
    f.save(mpath, overwrite=True, include_optimizer=False, save_format='h5')
        
        
if __name__ == '__main__':
    try:
        conf_path = sys.argv[1]
    except:
        print("USAGE: conf_file_gin")
        sys.exit(1)
        
    gin.parse_config_file(conf_path)
    
    name = basenameNoExt(conf_path)
    print("Name: ", name)
    
    setup(name)