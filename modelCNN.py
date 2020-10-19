import os
from functools import partial
import tensorflow as tf
import numpy as np
import architectureCNN as architecture

PAD = 0
    
    
#-----------------------------------------------------------------------------------------------
def make_model(hparams, DICT_SIZE, MAX_LEN):

    arch_id = hparams['arch']
    arch = architecture.archMap(arch_id)

    x = tf.keras.layers.Input(MAX_LEN, dtype=tf.int32)
    logits, z, att_ws = arch(x, MAX_LEN, DICT_SIZE, hparams)

    p = tf.nn.softmax(logits, 2)
    
    prediction = tf.argmax(p, 2, output_type=tf.int32)
    model = tf.keras.Model(inputs=x, outputs=[logits, p, prediction, z, att_ws])

    return model
#-----------------------------------------------------------------------------------------------
def loss_function(real, logits, z, hparams):    
    batch_size = hparams['batch_size']
    alpha = hparams['alpha']
    loss_type = hparams['loss_type']
    
    #if loss_type == 0:
    loss_class = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(real, logits)
    
    shape = batch_size, z.shape.as_list()[1]
    
    ztarget = tf.random.normal(shape)
    latent_reg = mmd_loss(z, ztarget) * alpha
    
    loss_ = loss_class + latent_reg
        
    return loss_
#----------------------------------------------------------------------------------------------- 

def make_train_predict(hparams, optimizer, DICT_SIZE, MAX_LEN):
    
    f = make_model(hparams, DICT_SIZE, MAX_LEN)
    
    print(f.summary())
    
    @tf.function
    def train_step(data):
        features, prediction_mask, labels = data
        with tf.GradientTape() as tape:
            # forward
            logits, p, prediction, z, _ = f(features, training=True)
            loss = loss_function(labels, logits, z, hparams)
                       
        gradients = tape.gradient(loss, f.trainable_variables)
        optimizer.apply_gradients(zip(gradients, f.trainable_variables))

        return loss, p, prediction

    @tf.function
    def predict_step(data):
        features, prediction_mask, labels = data
        # forward
        logits, p, prediction, z, _ = f(features, training=False)
        loss = loss_function(labels, logits, z, hparams)           
        
        return loss, p, prediction
    
    return f, train_step, predict_step


#----------______________-----------------________________-----------------

sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]

def compute_pairwise_distances(x, y):
    if not len(x.get_shape()) == len(y.get_shape()) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError('The number of features should be the same.')

    norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))

def gaussian_kernel_matrix(x, y, sigmas):
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
    dist = compute_pairwise_distances(x, y)
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))

def maximum_mean_discrepancy(x, y, kernel):
    with tf.name_scope('MaximumMeanDiscrepancy'):
        cost = tf.reduce_mean(kernel(x, x))
        cost += tf.reduce_mean(kernel(y, y))
        cost -= 2 * tf.reduce_mean(kernel(x, y))
        cost = tf.where(cost > 0, cost, 0, name='value')
    return cost

def mmd_loss(source_samples, target_samples, scope=None):
    """ from https://github.com/tensorflow/models/blob/master/research/domain_adaptation/domain_separation/losses.py """

    gaussian_kernel = partial(gaussian_kernel_matrix, sigmas=tf.constant(sigmas))

    loss_value = maximum_mean_discrepancy(source_samples, target_samples, kernel=gaussian_kernel)
    loss_value = tf.maximum(1e-4, loss_value)
    
    return loss_value