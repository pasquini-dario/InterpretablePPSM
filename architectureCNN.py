import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from architectureCNN import *
from input_pipeline import PAD

def archMap(arch_id):
    m = {
        0 : resnet_classic,
    }
    return m[arch_id]

EPSILON_NORM = 1e-6
layer_norm = lambda x: tf.keras.layers.LayerNormalization(epsilon=EPSILON_NORM)(x)


def ResBlockDeepBNK(inputs, dim, filter_size, activation, with_batch_norm=True, training=True):
    x = inputs

    dim_BNK = dim // 2

    if with_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Conv1D(dim_BNK, 3, padding='same')(x)

    if with_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Conv1D(dim_BNK, filter_size, padding='same')(x)

    if with_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Conv1D(dim, 3, padding='same')(x)

    return inputs + (0.3*x)


def ResBlockDeepBNK_separable(inputs, dim, filter_size, activation, with_batch_norm=True, training=True):
    x = inputs

    dim_BNK = dim // 2

    if with_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.SeparableConv1D(dim_BNK, 3, padding='same')(x)

    if with_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.SeparableConv1D(dim_BNK, filter_size, padding='same')(x)

    if with_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.SeparableConv1D(dim, 3, padding='same')(x)

    return inputs + (0.3*x)


BLOCKS_MAP = {
    0 : ResBlockDeepBNK,
    1 : ResBlockDeepBNK_separable
}

def resnet_backbone(x, max_len, vocab_size, hparams):
    
    batch_norm = True
    
    latent_size = hparams['latent_size']#128
    layer_dim = hparams['k']#128
    embedding_size = hparams['embedding_size']
    filter_size = hparams['filter_size']#3
    n_blocks = hparams['n_blocks'] #10
    block_type = hparams['block_type'] 
    activation = hparams['activation'] 
    
    block = BLOCKS_MAP[block_type]
    
    x = tf.keras.layers.Embedding(vocab_size, embedding_size, name="char_embedding")(x)
    #x = tf.one_hot(x, vocab_size)
    
    output_shape = max_len, vocab_size
    
    x = tf.keras.layers.Conv1D(layer_dim, filter_size, padding='same')(x)
    
    # encoder
    for i in range(n_blocks):
        x = block(x, layer_dim,  activation=activation, with_batch_norm=batch_norm, filter_size=filter_size)

    # latent space
    x = tf.keras.layers.Flatten()(x)
    z = tf.keras.layers.Dense(latent_size)(x)
    x = tf.keras.layers.Dense(output_shape[0]*layer_dim)(z)    
    x = tf.keras.layers.Reshape([output_shape[0], layer_dim])(x)

    # decoder
    for i in range(n_blocks):
        x = block(x, layer_dim, activation=activation, with_batch_norm=batch_norm,  filter_size=filter_size)
        
    return x, z


def resnet_classic(x, max_len, vocab_size, hparams):
    output_shape = max_len, vocab_size
    x, z = resnet_backbone(x, max_len, vocab_size, hparams)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(output_shape[0]*output_shape[1])(x)
    logits = tf.keras.layers.Reshape((output_shape[0], output_shape[1]))(x)
    
    return logits, z, z

    
