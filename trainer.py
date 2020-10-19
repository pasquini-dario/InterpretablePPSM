import tensorflow as tf
from tensorflow import keras as k
import numpy as np
import tensorflow_addons as tfa
from tensorboard.plugins.hparams import api as hp
import tqdm, os        
from glob import glob
from inference import Inference
from evaluate import getScore, readPC, rank

patience = 5
PROFILE_ITER = 1000
BATCH_EVAL = 2046

def flush_metric(iteration, metric, non_scalar=False):
    name = metric.name
    if non_scalar:
        value = metric.result()[0]
    else:
        value = metric.result()
    tf.summary.scalar(name, value, step=iteration)
    metric.reset_states()
    return (name, value)

flatten = lambda x: tf.reshape(x, (-1, 1)) 


class Trainer:
    def __init__(self,
                 f,
                 max_len,
                 cm,
                 train_step,
                 predict_step,
                 epochs,
                 train_batch,
                 home_tests,
                 optimizer,
                 log_train,
                 log_test,
                 test_num_steps,
                 log_freq,
                 hparams
                ):
        
        self.f = f
        self.max_len = max_len
        self.cm = cm
        self.home_tests = home_tests
        self.train_step = train_step
        self.predict_step = predict_step
        self.epochs = epochs
        
        self.train_batch = train_batch
        self.optimizer = optimizer
        self.log_freq = log_freq
        
        self.hparams = hparams
        
        self.test_num_steps = test_num_steps
        
        # early stopping
        self.top_score = None
        self.countdown = patience
        
        self.log_train = log_train
        
        self.train_summary_writer = train_summary_writer = tf.summary.create_file_writer(log_train)
        if log_test:
            self.test_summary_writer = test_summary_writer = tf.summary.create_file_writer(log_test)
        
        #check points
        checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            model=self.f,
        )

        self.ckpt_manager = tf.train.CheckpointManager(checkpoint, log_train, max_to_keep=patience)

        if self.ckpt_manager.latest_checkpoint:
            checkpoint.restore(self.ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!')


    def train_and_test(self, dataset, profile=False):
    
        # create metrics
        loss_m = tf.keras.metrics.Mean(name='loss')
        acc_m = tf.keras.metrics.Accuracy(name='accuracy')

        for i in range(self.epochs):
            print(f'Epoch {i}')
            for data in tqdm.tqdm(dataset):
                
                x, prediction_mask, y = data
                
                iteration = self.optimizer.iterations
                
                loss, p, prediction = self.train_step(data)
                    
                loss_m.update_state(loss)
                #acc_m.update_state(_y, _prediction)

                if tf.equal(iteration % self.log_freq, 0):

                    flush_metric(iteration, loss_m)
                    #flush_metric(iteration, acc_m)
                    
            if profile:
                print("END PROFILE")
                break 
                
            with self.test_summary_writer.as_default():
                    
                # application task test
                rc_scores = self.rankConfOnTestSets(BATCH_EVAL)
                avg_score = 0
                for name, score in rc_scores:
                    print(name, score)
                    avg_score += score[0]
                    tf.summary.scalar(f'WRankCof_{name}', score[0], step=i+1)
                avg_score = avg_score / len(rc_scores)
                
                tf.summary.scalar(f'WRankCof_avg', avg_score, step=i+1)
                
                #-----------
                if self.early_stopping(-avg_score):
                    print(f"Early-stop epoch-{i}")
                    break

                
                    
    
    
    def rankConfOnTestSets(self, batch_size):
        inf = Inference(self.f, self.cm, self.max_len, batch_size)

        scores = []
        paths = glob(self.home_tests)
        for path in paths:
            name = os.path.basename(path).split('-')[0]
            print(name)
            X, F = readPC(path, self.max_len-1, encoding='ascii')
            R = rank(F)
            R = np.array(R)

            # apply model
            UP = inf.applyBatch(X, INCLUDE_END_SYMBOL=self.hparams['append_end'])
            UP = np.array(UP)
            score = getScore(UP, R)
            scores += [(name, score)]
        
        return scores
    
    def early_stopping(self, test_score):
        print(test_score)
        if self.top_score is None or test_score < self.top_score:
            print("New Best", test_score)
            self.top_score = test_score
            self.countdown = patience
            
            # Save checkpoint
            ckpt_save_path = self.ckpt_manager.save()
            print ('Saving checkpoint at {}'.format(ckpt_save_path))
            
            return False
        
        self.countdown -= 1
        if self.countdown == 0:
            print("UTB", test_score, self.top_score)
            return True


    def __call__(self, profile_run=False):
        if profile_run:
            with tf.profiler.experimental.Profile(self.log_train):
                self.train(self.train_batch.take(PROFILE_ITER), True)
        else:
            with self.train_summary_writer.as_default():
                self.train_and_test(self.train_batch)
                hp.hparams(self.hparams)

                