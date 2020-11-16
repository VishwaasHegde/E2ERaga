from __future__ import division
from __future__ import print_function

import os
import re
import sys
import math
from data_generator import DataGenerator
from scipy.io import wavfile
import numpy as np
from numpy.lib.stride_tricks import as_strided
import tensorflow as tf
import tensorflow_transform as tft
# tf.config.run_functions_eagerly(True)
# tf.compat.v1.disable_eager_execution()
from tensorflow.keras.layers import Input, Reshape, Conv2D, BatchNormalization, Softmax, Conv1D, Bidirectional
from tensorflow.keras.layers import MaxPool2D, Dropout, Permute, Flatten, Dense, MaxPool1D, AvgPool1D, Bidirectional, LSTM
from tensorflow.keras.models import Model
import librosa
import librosa.display
from encoder_2 import Encoder
import pyhocon
import os
import json
import pandas as pd
from collections import defaultdict
from tensorflow.keras.callbacks import ModelCheckpoint

import encoder_3

import matplotlib.pyplot as plt
# store as a global variable, since we only support a few models for now
from data_generator import DataGenerator

models = {
    'tiny': None,
    'small': None,
    'medium': None,
    'large': None,
    'full': None
}

# the model is trained on 16kHz audio
# model_srate = 16000
# max_batch_size = 3000
# sequence_length = 200
# n_labels = 30
# config = pyhocon.ConfigFactory.parse_file("crepe/experiments.conf")['test']

def build_and_load_model(config, task ='raga'):
    """
    Build the CNN model and load the weights

    Parameters
    ----------
    model_capacity : 'tiny', 'small', 'medium', 'large', or 'full'
        String specifying the model capacity, which determines the model's
        capacity multiplier to 4 (tiny), 8 (small), 16 (medium), 24 (large),
        or 32 (full). 'full' uses the model size specified in the paper,
        and the others use a reduced number of filters in each convolutional
        layer, resulting in a smaller model that is faster to evaluate at the
        cost of slightly reduced pitch estimation accuracy.

    Returns
    -------
    model : tensorflow.keras.models.Model
        The pre-trained keras model loaded in memory
    """
    model_capacity = config['model_capacity']
    model_srate = config['model_srate']
    hop_size = int(config['hop_size']*model_srate)
    sequence_length = int((config['sequence_length']*model_srate - 1024)/hop_size) + 1
    drop_rate = config['drop_rate']
    cutoff = config['cutoff']
    n_frames = 1 + int((model_srate * cutoff - 1024) / hop_size)
    n_seq = int(n_frames // sequence_length)

    n_labels = config['n_labels']

    note_dim = config['note_dim']

    # x = Input(shape=(1024,), name='input2', dtype='float32')
    x_batch = Input(shape=(None, 1024), name='x_input', dtype='float32')
    tonic_batch = Input(shape=(60,), name='tonic_input', dtype='float32')
    pitches_batch = Input(shape=(None, 360), name='pitches_input', dtype='float32')
    transpose_by_batch = Input(shape=(), name='transpose_input', dtype='int32')

    x = x_batch[0]
    tonic_input = tf.expand_dims(tonic_batch[0], 0)
    pitches = pitches_batch[0]
    transpose_by = transpose_by_batch[0]

    y, note_emb = get_pitch_emb(x, n_seq, n_frames, model_capacity)
    pitch_model = Model(inputs=[x_batch], outputs=y)

    if task=='pitch':
        return pitch_model
    pitch_model.load_weights('model/hindustani_pitch_model.hdf5', by_name=True)

    note_emb = reduce_note_emb_dimensions(note_emb, note_dim)

    red_y = tf.reshape(pitches, [-1, 6, 60])
    red_y = tf.reduce_mean(red_y, axis=1)  # (None, 60)

    # transpose_by = tf.random.uniform(shape=(), minval=0, maxval=60, dtype=tf.int32)
    red_y = tf.roll(red_y, -transpose_by, axis=1)
    # tonic_input = tf.roll(tonic_input, -transpose_by, axis=1)

    #Tonic
    tonic_emb = get_tonic_emb(red_y, note_emb, note_dim, drop_rate)

    # tonic_logits = tf.reduce_mean(tf.multiply(note_emb, tf.tile(tonic_emb, [60,1])), axis=1, keepdims=True)
    # tonic_logits = tf.nn.sigmoid(tf.transpose(tonic_logits))
    tonic_logits = Dense(60, activation='sigmoid', name='tonic')(tonic_emb)

    tonic_model = Model(inputs=[pitches_batch, transpose_by_batch], outputs=tonic_logits)
    tonic_model.summary()
    tonic_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    tonic_model.load_weights('model/hindustani_raga_model.hdf5', by_name=True)

    if task== 'tonic':
        return tonic_model

    # for layer in tonic_model.layers:
    #     layer.trainable = False

    # tonic_logits_masked = tonic_logits[0]
    tonic_logits_masked = tonic_input[0]
    tonic_logits_pad = tf.pad(tonic_logits_masked, [[5,5]])
    tonic_logits_argmax = tf.cast(tf.argmax(tonic_logits_pad), tf.int32)
    tonic_indices = tf.range(70)
    lower_limit = tf.less(tonic_indices, tonic_logits_argmax-4)
    upper_limit = tf.greater(tonic_indices, tonic_logits_argmax + 5)
    tonic_logits_mask = 1 - tf.cast(tf.logical_or(lower_limit, upper_limit), tf.float32)
    tonic_logits_mask = tonic_logits_mask[5:-5]
    tonic_logits_masked = tf.multiply(tonic_logits_masked, tonic_logits_mask)
    tonic_logits_masked = tonic_logits_masked/tf.reduce_sum(tonic_logits_masked)
    tonic_logits_masked = tf.expand_dims(tonic_logits_masked, 0)

    # rag_emb = get_rag_emb_1(red_y, tonic_input, note_emb, note_dim, drop_rate)

    # rag_emb = get_rag_emb_2(red_y, tonic_logits, note_emb, note_dim, drop_rate)
    # rag_emb = get_rag_emb_2(red_y, tonic_input, note_emb, note_dim, drop_rate)
    rag_emb = get_rag_emb_2(red_y, tonic_logits_masked, note_emb, note_dim, drop_rate)
    # raga_logits = Dense(n_labels, activation='softmax', name='raga')(tf.expand_dims(pitches[0],0))
    raga_logits = Dense(n_labels, activation='softmax', name='raga')(rag_emb)

    loss_weights = config['loss_weights']

    # rag_model = Model(inputs=[pitches_batch, tonic_batch], outputs=[raga_logits])
    # rag_model = Model(inputs=[pitches_batch, transpose_by_batch], outputs=[tonic_logits, raga_logits])
    # rag_model = Model(inputs=[pitches_batch, transpose_by_batch], outputs=[tonic_logits, raga_logits])
    rag_model = Model(inputs=[pitches_batch, tonic_batch, transpose_by_batch], outputs=[raga_logits])

    # rag_model = Model(inputs=[x_batch], outputs=[tonic_logits, raga_logits])
    # rag_model.compile(loss={'tonic': 'binary_crossentropy', 'raga': 'categorical_crossentropy'},
    #               optimizer='adam', metrics={'tonic': 'categorical_accuracy', 'raga': 'accuracy'}, loss_weights={'tonic': loss_weights[0], 'raga': loss_weights[1]})
    rag_model.compile(loss={'raga': 'categorical_crossentropy'},
                  optimizer='adam', metrics={'raga': 'accuracy'}, loss_weights={'raga': loss_weights[1]})

    # rag_model.load_weights('model/hindustani_raga_model.hdf5', by_name=True)
    # rag_model = Model(inputs=[x_batch, chroma_batch, energy_batch, tonic_batch], outputs=[raga_logits])
    # rag_model.compile(loss={'raga': 'categorical_crossentropy'},
    #                   optimizer='adam', metrics={'raga': 'accuracy'}, loss_weights={'raga': loss_weights[1]})
    rag_model.summary()
    # rag_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return rag_model

def get_pitch_emb(x, n_seq, n_frames, model_capacity):
    capacity_multiplier = {
        'tiny': 4, 'small': 8, 'medium': 16, 'large': 24, 'full': 32
    }[model_capacity]
    layers = [1, 2, 3, 4, 5, 6]
    filters = [n * capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
    widths = [512, 64, 64, 64, 64, 64]
    strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

    z = []
    layers_cache = []
    for i in range(n_seq - 1):
        x_pitch = x[int(i * n_frames / n_seq):int((i + 1) * n_frames / n_seq)]

        if i == 0:
            res = Reshape(target_shape=(1024, 1, 1), name='input-reshape')
            layers_cache.append(res)
            conv_layers = []
        else:
            res = layers_cache[0]
            conv_layers = layers_cache[1]
        y = res(x_pitch)
        m = 0
        for l, f, w, s in zip(layers, filters, widths, strides):
            if i == 0:
                conv_1 = Conv2D(f, (w, 1), strides=s, padding='same',
                                activation='relu', name="conv%d" % l, trainable=False)
                bn_1 = BatchNormalization(name="conv%d-BN" % l)
                mp_1 = MaxPool2D(pool_size=(2, 1), strides=None, padding='valid',
                                 name="conv%d-maxpool" % l, trainable=False)
                do_1 = Dropout(0.25, name="conv%d-dropout" % l)
                conv_layers.append([conv_1, bn_1, mp_1, do_1])
            else:
                conv_1, bn_1, mp_1, do_1 = conv_layers[m]

            y = conv_1(y)
            y = bn_1(y)
            y = mp_1(y)
            y = do_1(y)
            m += 1

        if i == 0:
            den = Dense(360, activation='sigmoid', name="classifier", trainable=False)
            per = Permute((2, 1, 3))
            flat = Flatten(name="flatten")
            layers_cache.append(conv_layers)
            layers_cache.append(den)
            layers_cache.append(per)
            layers_cache.append(flat)
        else:
            den = layers_cache[2]
            per = layers_cache[3]
            flat = layers_cache[4]

        y = per(y)
        y = flat(y)
        y = den(y)
        z.append(y)
    y = tf.concat(z, axis=0)

    return y, den.weights[0]

def reduce_note_emb_dimensions(emb, note_dim):
    note_emb = emb
    note_emb = tf.reduce_mean(tf.reshape(note_emb, [-1, 6, 60]), axis=1)
    note_emb = tf.transpose(note_emb, name='note_emb')  # 60,note_emb
    note_emb = tf.tile(note_emb, [tf.cast(tf.math.ceil(tf.shape(note_emb)[1] / 60), tf.int32), 1])

    singular_values, u, _ = tf.linalg.svd(note_emb)
    sigma = tf.linalg.diag(singular_values)
    sigma = tf.slice(sigma, [0, 0], [tf.shape(note_emb)[-1], note_dim])
    pca = tf.matmul(u, sigma)
    note_emb = pca[:60, :]

    return note_emb

def get_tonic_emb(red_y, note_emb, note_dim, drop_rate=0.2):
    tonic_cnn = get_tonic_from_cnn(red_y, note_emb, note_dim, drop_rate)  # (60, 32)
    tonic_rnn = get_tonic_from_rnn(red_y, note_emb, note_dim, drop_rate)

    f = tf.nn.sigmoid(Dense(note_dim)(tf.concat([tonic_cnn, tonic_rnn], axis=1)))
    # f= 0
    tonic_emb = f * tonic_cnn + (1 - f) * tonic_rnn
    return tonic_emb

# @tf.function
def get_tonic_from_cnn(red_y, note_emb, note_dim, drop_rate=0.2):

    diag_tf = tf.reduce_mean(red_y, axis=0)
    diag_tf = AvgPool1D(pool_size=2, strides=1, padding='same')(tf.expand_dims(tf.expand_dims(diag_tf, 0), 2))[0, :, 0]
    diag_tf_p = tf.roll(diag_tf, 1, 0)
    diag_tf_n = tf.roll(diag_tf, -1, 0)
    diag_tf_1 = tf.less_equal(diag_tf_p, diag_tf)
    diag_tf_2 = tf.less_equal(diag_tf_n, diag_tf)
    diag_tf_3 = tf.logical_and(diag_tf_1, diag_tf_2)
    diag_tf_3 = tf.cast(diag_tf_3, tf.float32)
    diag_tf_4 = tf.multiply(2 * diag_tf - diag_tf_p - diag_tf_n, diag_tf_3)
    diag_tf_3 = tf.multiply(diag_tf, diag_tf_3)

    diag_tf_3 = min_max_scale(diag_tf_3)
    diag_tf_4 = min_max_scale(diag_tf_4)

    pitch_dot = tf.reduce_mean(tf.multiply(note_emb, tf.tile(tf.expand_dims(note_emb[0],0),[60,1])), axis=1)
    pitch_dot = min_max_scale(pitch_dot)
    hist_cc = tf.transpose(tf.stack([diag_tf,diag_tf_3,diag_tf_4,pitch_dot]))

    hist_cc = tf.expand_dims(hist_cc,0)
    z = Conv1D(filters=128, kernel_size=5, strides=1,padding='valid', activation='relu')(hist_cc)
    z = BatchNormalization()(z)
    z = MaxPool1D(pool_size=2)(z)
    z = Dropout(drop_rate)(z)
    z = Conv1D(filters=256, kernel_size=3, strides=1,padding='valid', activation='relu')(z)
    z = BatchNormalization()(z)
    z = MaxPool1D(pool_size=2)(z)
    z = Dropout(drop_rate)(z)
    z = Flatten(name='tonic_flat')(z)
    tonic_emb = Dense(note_dim, activation='relu')(z)

    return tonic_emb

def get_tonic_from_rnn(red_y, note_emb, note_dim, drop_rate=0.2):
    diag_tf = tf.reduce_mean(red_y, axis=0)
    diag_tf = AvgPool1D(pool_size=2, strides=1, padding='same')(tf.expand_dims(tf.expand_dims(diag_tf, 0), 2))[0, :, 0]
    diag_tf_p = tf.roll(diag_tf, 1, 0)
    diag_tf_n = tf.roll(diag_tf, -1, 0)
    diag_tf_1 = tf.less_equal(diag_tf_p, diag_tf)
    diag_tf_2 = tf.less_equal(diag_tf_n, diag_tf)
    diag_tf_3 = tf.logical_and(diag_tf_1, diag_tf_2)
    diag_tf_3 = tf.cast(diag_tf_3, tf.float32)
    diag_tf_3_tile = tf.tile(tf.expand_dims(diag_tf_3, 0), [tf.shape(red_y)[0],1])

    red_y_am = tf.argmax(red_y, axis=1)
    red_y_am = tf.one_hot(red_y_am,60)
    red_y_am = tf.multiply(diag_tf_3_tile, red_y_am)
    red_y_am_nz = tf.reduce_sum(red_y_am, axis=1)
    red_y_am_nz = tf.where(red_y_am_nz)[:,0]
    red_y_am = tf.gather(red_y_am, red_y_am_nz)
    red_y_am = tf.argmax(red_y_am, 1)
    red_y_am = get_ndms(red_y_am)
    encoding = get_rag_from_rnn(red_y_am, note_emb, note_dim, drop_rate)
    encoding = Dense(note_dim, activation='relu')(encoding)
    return encoding

def get_rag_emb_1(red_y, tonic_input, note_emb, note_dim, drop_rate=0.25):
    transpose_by = tf.argmax(tonic_input, axis=1)[0]
    red_y = tf.roll(red_y, -transpose_by, axis=1)
    diag_tf = tf.reduce_mean(red_y, axis=0)
    diag_tf = AvgPool1D(pool_size=2, strides=1, padding='same')(tf.expand_dims(tf.expand_dims(diag_tf, 0), 2))[0, :, 0]
    diag_tf_p = tf.roll(diag_tf, 1, 0)
    diag_tf_n = tf.roll(diag_tf, -1, 0)
    diag_tf_1 = tf.less_equal(diag_tf_p, diag_tf)
    diag_tf_2 = tf.less_equal(diag_tf_n, diag_tf)
    diag_tf_3 = tf.logical_and(diag_tf_1, diag_tf_2)
    diag_tf_3 = tf.cast(diag_tf_3, tf.float32)
    diag_tf_3_tile = tf.tile(tf.expand_dims(diag_tf_3, 0), [tf.shape(red_y)[0],1])
    diag_tf_4 = tf.multiply(2 * diag_tf - diag_tf_p - diag_tf_n, diag_tf_3)
    diag_tf_3 = tf.multiply(diag_tf, diag_tf_3)

    red_y_am = tf.argmax(red_y, axis=1)
    red_y_am = tf.one_hot(red_y_am,60)
    red_y_am = tf.multiply(diag_tf_3_tile, red_y_am)
    red_y_am_nz = tf.reduce_sum(red_y_am, axis=1)
    red_y_am_nz = tf.where(red_y_am_nz)[:,0]
    red_y_am = tf.gather(red_y_am, red_y_am_nz)
    red_y_am = tf.argmax(red_y_am, 1)
    red_y_am = get_ndms(red_y_am)
    diag_tf_3 = min_max_scale(diag_tf_3)
    diag_tf_4 = min_max_scale(diag_tf_4)

    note_emb_add = note_emb
    encoding = get_rag_from_rnn(red_y_am, note_emb_add, note_dim, drop_rate)
    pitch_dot = tf.reduce_mean(tf.multiply(note_emb, tf.tile(tf.expand_dims(note_emb[0],0),[60,1])), axis=1)
    pitch_dot = min_max_scale(pitch_dot)
    hist_cc = tf.transpose(tf.stack([diag_tf,diag_tf_3,diag_tf_4,pitch_dot]))

    hist_cc = tf.expand_dims(hist_cc,0)
    z = Conv1D(filters=64, kernel_size=5, strides=1,padding='valid', activation='relu')(hist_cc)
    z = BatchNormalization()(z)
    z = MaxPool1D(pool_size=2)(z)
    z = Dropout(drop_rate)(z)
    z = Conv1D(filters=128, kernel_size=3, strides=1,padding='valid', activation='relu')(z)
    z = BatchNormalization()(z)
    z = MaxPool1D(pool_size=2)(z)
    z = Dropout(drop_rate)(z)
    z = Flatten(name='raga_flat')(z)
    z = Dense(2 * note_dim, activation='relu')(z)

    f = tf.nn.sigmoid(Dense(2*note_dim)(tf.concat([z, encoding], axis=1)))
    raga_emb = f*z+(1-f)*encoding
    # raga_emb = encoding
    return raga_emb

def get_rag_emb_2(red_y, tonic_logits, note_emb, note_dim, drop_rate=0.25):

    diag_tf = tf.reduce_mean(red_y, axis=0)
    diag_tf = AvgPool1D(pool_size=2, strides=1, padding='same')(tf.expand_dims(tf.expand_dims(diag_tf, 0), 2))[0, :, 0]
    diag_tf_p = tf.roll(diag_tf, 1, 0)
    diag_tf_n = tf.roll(diag_tf, -1, 0)
    diag_tf_1 = tf.less_equal(diag_tf_p, diag_tf)
    diag_tf_2 = tf.less_equal(diag_tf_n, diag_tf)
    diag_tf_3 = tf.logical_and(diag_tf_1, diag_tf_2)
    diag_tf_3 = tf.cast(diag_tf_3, tf.float32)
    diag_tf_3_tile = tf.tile(tf.expand_dims(diag_tf_3, 0), [tf.shape(red_y)[0],1])
    diag_tf_4 = tf.multiply(2 * diag_tf - diag_tf_p - diag_tf_n, diag_tf_3)
    diag_tf_3 = tf.multiply(diag_tf, diag_tf_3)
    diag_tf_3 = min_max_scale(diag_tf_3)
    diag_tf_4 = min_max_scale(diag_tf_4)

    red_y_am = tf.argmax(red_y, axis=1)
    red_y_am = tf.one_hot(red_y_am,60)
    red_y_am = tf.multiply(diag_tf_3_tile, red_y_am)
    red_y_am_nz = tf.reduce_sum(red_y_am, axis=1)
    red_y_am_nz = tf.where(red_y_am_nz)[:,0]
    red_y_am = tf.gather(red_y_am, red_y_am_nz)
    red_y_am = tf.argmax(red_y_am, 1)
    red_y_am = get_ndms(red_y_am)

    hist_cc = tf.transpose(tf.stack([diag_tf, diag_tf_3, diag_tf_4]))
    hist_cc_all = []
    red_y_ndms = []
    for i in range(60):
        hist_cc_trans = tf.roll(hist_cc, -i, axis=0)
        red_y_am_trans = tf.math.mod(60+red_y_am-i,60)
        hist_cc_all.append(hist_cc_trans)
        red_y_ndms.append(red_y_am_trans)
    hist_cc_all = tf.stack(hist_cc_all)
    red_y_ndms = tf.stack(red_y_ndms)
    encoding = get_rag_from_rnn(red_y_ndms, note_emb, note_dim, drop_rate)

    z = Conv1D(filters=128, kernel_size=5, strides=1,padding='valid', activation='relu')(hist_cc_all)
    z = BatchNormalization()(z)
    z = MaxPool1D(pool_size=2)(z)
    z = Dropout(drop_rate)(z)
    z = Conv1D(filters=192, kernel_size=3, strides=1,padding='valid', activation='relu')(z)
    z = BatchNormalization()(z)
    z = MaxPool1D(pool_size=2)(z)
    z = Dropout(drop_rate)(z)
    z = Flatten(name='raga_flat')(z)
    # z = tf.concat([z, diag_tf_3_den], axis=1)
    z = Dense(2 * note_dim, activation='relu')(z)

    f = tf.nn.sigmoid(Dense(2*note_dim)(tf.concat([z, encoding], axis=1)))

    raga_emb = f*z+(1-f)*encoding
    # raga_emb = z

    raga_emb = tf.multiply(raga_emb, tf.transpose(tonic_logits))
    raga_emb = tf.reduce_sum(raga_emb, axis=0, keepdims=True)
    raga_emb = raga_emb/tf.reduce_sum(tonic_logits)


    return raga_emb


def get_rag_from_rnn(red_y_am, note_emb_add, note_dim, dropout):
    embs = tf.gather(note_emb_add, red_y_am)
    if len(embs.shape)==2:
        embs = tf.expand_dims(embs, 0)
    rnn_1 = Bidirectional(LSTM(note_dim, return_sequences=True, recurrent_dropout=dropout))(embs)
    rnn_1 = Dropout(dropout)(rnn_1)
    rnn_1 = Bidirectional(LSTM(note_dim))(rnn_1)
    # rnn_1 = tf.expand_dims(rnn_1[0],0)

    return Dense(2 * note_dim, activation='relu')(rnn_1)

def min_max_scale(y):
    y_min = tf.reduce_min(y)
    return (y - y_min)/(tf.reduce_max(y)-y_min)

def ffnn(inputs, hidden_size, drop_rate=0.4):
    x = inputs
    for hs in hidden_size:
        den = Dense(hs, activation='relu')(x)
        x = Dropout(drop_rate)(den)
    return x

def get_ndms(arg_y):
    # red_y = tf.random.uniform(shape=(100,), maxval=60, dtype=tf.int32)
    # red_y  = tf.one_hot(red_y,60)

    arg_y = tf.concat([[0.],tf.cast(arg_y, tf.float32)], axis=-1) #None+1


    arg_y_shifted = tf.roll(arg_y,-1, axis=-1) #1,None+1

    mask = tf.cast(tf.not_equal(arg_y, arg_y_shifted), tf.float32)  #1,None+1
    mask = tf.where(mask)[:,0]
    uni_seq_notes = tf.gather(arg_y_shifted, mask)
    uni_seq_notes = tf.cast(uni_seq_notes, tf.int32)
    return uni_seq_notes
    



def output_path(file, suffix, output_dir):
    """
    return the output path of an output file corresponding to a wav file
    """
    path = re.sub(r"(?i).wav$", suffix, file)
    if output_dir is not None:
        path = os.path.join(output_dir, os.path.basename(path))
    return path


def to_local_average_cents(salience, center=None):
    """
    find the weighted average cents near the argmax bin
    """

    if not hasattr(to_local_average_cents, 'cents_mapping'):
        # the bin number-to-cents mapping
        to_local_average_cents.cents_mapping = (
                np.linspace(0, 7180, 360) + 1997.3794084376191)

    if salience.ndim == 1:
        if center is None:
            center = int(np.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience = salience[start:end]
        product_sum = np.sum(
            salience * to_local_average_cents.cents_mapping[start:end])
        weight_sum = np.sum(salience)
        return product_sum / weight_sum
        # product_sum = np.sum(
        #     salience * to_local_average_cents.cents_mapping)
        # return product_sum
    if salience.ndim == 2:
        return np.array([to_local_average_cents(salience[i, :]) for i in
                         range(salience.shape[0])])

    raise Exception("label should be either 1d or 2d ndarray")


def to_viterbi_cents(salience):
    """
    Find the Viterbi path using a transition prior that induces pitch
    continuity.
    """
    from hmmlearn import hmm

    # uniform prior on the starting pitch
    starting = np.ones(360) / 360

    # transition probabilities inducing continuous pitch
    xx, yy = np.meshgrid(range(360), range(360))
    transition = np.maximum(12 - abs(xx - yy), 0)
    transition = transition / np.sum(transition, axis=1)[:, None]

    # emission probability = fixed probability for self, evenly distribute the
    # others
    self_emission = 0.1
    emission = (np.eye(360) * self_emission + np.ones(shape=(360, 360)) *
                ((1 - self_emission) / 360))

    # fix the model parameters because we are not optimizing the model
    model = hmm.MultinomialHMM(360, starting, transition)
    model.startprob_, model.transmat_, model.emissionprob_ = \
        starting, transition, emission

    # find the Viterbi path
    observations = np.argmax(salience, axis=1)
    path = model.predict(observations.reshape(-1, 1), [len(observations)])

    return np.array([to_local_average_cents(salience[i, :], path[i]) for i in
                     range(len(observations))])


def get_activation(audio, sr, model_capacity='full', center=True, step_size=10,
                   verbose=1):
    """

    Parameters
    ----------
    audio : np.ndarray [shape=(N,) or (N, C)]
        The audio samples. Multichannel audio will be downmixed.
    sr : int
        Sample rate of the audio samples. The audio will be resampled if
        the sample rate is not 16 kHz, which is expected by the model.
    model_capacity : 'tiny', 'small', 'medium', 'large', or 'full'
        String specifying the model capacity; see the docstring of
        :func:`~crepe.core.build_and_load_model`
    center : boolean
        - If `True` (default), the signal `audio` is padded so that frame
          `D[:, t]` is centered at `audio[t * hop_length]`.
        - If `False`, then `D[:, t]` begins at `audio[t * hop_length]`
    step_size : int
        The step size in milliseconds for running pitch estimation.
    verbose : int
        Set the keras verbosity mode: 1 (default) will print out a progress bar
        during prediction, 0 will suppress all non-error printouts.

    Returns
    -------
    activation : np.ndarray [shape=(T, 360)]
        The raw activation matrix
    """
    config = pyhocon.ConfigFactory.parse_file("crepe/experiments.conf")['raga']
    model = build_and_load_model(config)
    if len(audio.shape) == 2:
        audio = audio.mean(1)  # make mono
    audio = audio.astype(np.float32)

    if sr != model_srate:
        # resample audio if necessary
        from resampy import resample
        audio = resample(audio, sr, model_srate)
    chroma = get_chroma(audio, model_srate)
    # pad so that frames are centered around their timestamps (i.e. first frame
    # is zero centered).

    if center:
        audio = np.pad(audio, 512, mode='constant', constant_values=0)

    # make 1024-sample frames of the audio with hop length of 10 milliseconds
    hop_length = int(model_srate * step_size / 1000)
    n_frames = 1 + int((len(audio) - 1024) / hop_length)
    frames = as_strided(audio, shape=(1024, n_frames),
                        strides=(audio.itemsize, hop_length * audio.itemsize))
    frames = frames.transpose().copy()
    # frames = np.expand_dims(1, frames)
    energy = (audio-np.mean(audio))/np.std(audio)
    energy = np.square(energy)
    energy_frames = as_strided(energy, shape=(1024, n_frames),
                        strides=(energy.itemsize, hop_length * energy.itemsize))
    energy_frames = energy_frames.transpose().copy()
    energy_frames = np.mean(energy_frames, axis=1)
    energy_frames = (energy_frames-np.mean(energy_frames))/np.std(energy_frames)

    frames = (frames - np.mean(frames, axis=1))/np.std(frames, axis=1)

    frames, energy_frames, mask = pad_frames(frames, energy_frames, sequence_length)
    frames = np.array([frames])
    mask = np.array([mask])
    chroma = np.array([chroma])
    energy_frames = np.array([energy_frames])

    # normalize each frame -- this is expected by the model
    # frames -= np.mean(frames, axis=1)[:, np.newaxis]
    # frames /= np.std(frames, axis=1)[:, np.newaxis]

    # run prediction and convert the frequency bin weights to Hz
    # print(tonic_model.predict(frames, verbose=verbose, batch_size = max_batch_size))
    # print(sil_model.predict(frames, verbose=verbose, batch_size=max_batch_size))
    # en = energy_model.predict(frames, verbose=verbose, batch_size=32 * 7 *3)
    # plt.plot(np.arange(0,len(energy_frames)), energy_frames)
    # plt.show()

    return model.predict([frames, mask,chroma, energy_frames], verbose=verbose, batch_size=max_batch_size)


def pad_frames(frames, sequence_length, energy_frames, step_size=10):
    padded_length = sequence_length * np.ceil(len(frames) / sequence_length)
    add_length = int(padded_length) - frames.shape[0]
    add_frames = np.zeros([add_length, 1024])-1
    padded_frames = np.concatenate([frames, add_frames], axis=0)

    mask = np.ones(frames.shape[0])
    mask = np.concatenate([mask, np.zeros(add_length)], axis=0)

    energy_frames = np.concatenate([energy_frames, np.zeros(add_length)], axis=0)

    int(max_batch_size*step_size / 1000)
    return padded_frames, energy_frames, mask

def train(task, tradition):
    if task=='tonic':
        train_tonic(tradition)
    elif task == 'raga':
        # tonic_model_path = train_tonic(tradition)
        raga_model_path = 'model/{}_raga_model.hdf5'.format(tradition)

        config = pyhocon.ConfigFactory.parse_file("experiments.conf")[task]
        training_generator = DataGenerator(task, tradition, 'train', config)
        validation_generator = DataGenerator(task, tradition, 'validate', config)
        model = build_and_load_model(config, task)
        # model.load_weights(tonic_model_path, by_name=True)
        # model.load_weights('model/model-full.h5', by_name=True)
        # model.fit(generator)
        # model.fit(x=training_generator,
        #                     validation_data=validation_generator, verbose=2, epochs=15, shuffle=True, batch_size=1)

        checkpoint = ModelCheckpoint(raga_model_path, monitor='loss', verbose=1,
                                     save_best_only=True, mode='auto', period=1)
        model.fit_generator(generator=training_generator,
                            validation_data=validation_generator, verbose=1, epochs=15, shuffle=False, callbacks=[checkpoint])

def train_tonic(tradition):
    task = 'tonic'
    model_path = 'model/{}_tonic_model.hdf5'.format(tradition)
    # if os.path.exists(model_path):
    #     return model_path

    config = pyhocon.ConfigFactory.parse_file("experiments.conf")[task]
    training_generator = DataGenerator(task, tradition, 'train', config)
    validation_generator = DataGenerator(task, tradition, 'validate', config)
    model = build_and_load_model(config, task)
    # model.load_weights('model/model-large.h5', by_name=True)
    # model.fit(generator)
    # model.fit(x=training_generator,
    #                     validation_data=validation_generator, verbose=2, epochs=15, shuffle=True, batch_size=1)
    checkpoint = ModelCheckpoint(model_path, monitor='loss', verbose=1,
                                 save_best_only=True, mode='auto', period=1)
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator, verbose=1, epochs=15, shuffle=True, callbacks=[checkpoint])

    return model_path

def test(task, tradition):
    config = pyhocon.ConfigFactory.parse_file("experiments.conf")[task]
    test_generator = DataGenerator(task, tradition, 'test', config, random=False)
    model = build_and_load_model(config, task)
    # model.fit(generator)
    # model.fit(x=training_generator,
    #                     validation_data=validation_generator, verbose=2, epochs=15, shuffle=True, batch_size=1)
    # model.load_weights('model/hindustani_tonic_model.hdf5', by_name=True)
    # model.load_weights('model/model-full.h5'.format(tradition, 'tonic'), by_name=True)
    # model.load_weights('model/{}_{}_model.hdf5'.format(tradition, 'tonic'), by_name=True)
    model.load_weights('model/hindustani_raga_model.hdf5', by_name=True)
    p = model.predict_generator(test_generator, verbose=1)
    # print(p[0, :, :, 0])
    # plt.imshow(p[1,:,:,1], cmap='hot', interpolation='nearest')
    # plt.show()


    # plt.imshow(p[1, :, :, 1], cmap='hot', interpolation='nearest')
    # plt.show()
    # plt.imshow(p[1, :, :, 2], cmap='hot', interpolation='nearest')
    # plt.show()
    # plt.imshow(p[1, :, :, 3], cmap='hot', interpolation='nearest')
    # plt.show()

    # print(p)
    # print(np.argmax(p[0]))
    print(np.argmax(p, axis=1))
    # cents = to_local_average_cents(p)
    # frequency = 10 * 2 ** (cents / 1200)
    #
    # for f in frequency:
    #     print(f)
    # print('pred', frequency)

def get_chroma(audio, sr):
    # logC = librosa.amplitude_to_db(np.abs(C))
    # plt.figure(figsize=(15, 5))
    # librosa.display.specshow(logC, sr=sr, x_axis='time', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')

    # hop_length = 512
    # chromagram = librosa.feature.chroma_cqt(audio, sr=sr, hop_length=hop_length)
    # plt.figure(figsize=(15, 5))
    # librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')

    hop_length = 512
    chromagram = librosa.feature.chroma_cens(audio, sr=sr, hop_length=hop_length, n_chroma=60, bins_per_octave=60)
    # plt.figure(figsize=(15, 5))
    # librosa.display.specshow(chromagram, sr=sr,x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm',bins_per_octave=60)
    return chromagram

def predict(audio, sr, model_capacity='full',
            viterbi=False, center=True, step_size=10, verbose=1):
    """
    Perform pitch estimation on given audio

    Parameters
    ----------
    audio : np.ndarray [shape=(N,) or (N, C)]
        The audio samples. Multichannel audio will be downmixed.
    sr : int
        Sample rate of the audio samples. The audio will be resampled if
        the sample rate is not 16 kHz, which is expected by the model.
    model_capacity : 'tiny', 'small', 'medium', 'large', or 'full'
        String specifying the model capacity; see the docstring of
        :func:`~crepe.core.build_and_load_model`
    viterbi : bool
        Apply viterbi smoothing to the estimated pitch curve. False by default.
    center : boolean
        - If `True` (default), the signal `audio` is padded so that frame
          `D[:, t]` is centered at `audio[t * hop_length]`.
        - If `False`, then `D[:, t]` begins at `audio[t * hop_length]`
    step_size : int
        The step size in milliseconds for running pitch estimation.
    verbose : int
        Set the keras verbosity mode: 1 (default) will print out a progress bar
        during prediction, 0 will suppress all non-error printouts.

    Returns
    -------
    A 4-tuple consisting of:

        time: np.ndarray [shape=(T,)]
            The timestamps on which the pitch was estimated
        frequency: np.ndarray [shape=(T,)]
            The predicted pitch values in Hz
        confidence: np.ndarray [shape=(T,)]
            The confidence of voice activity, between 0 and 1
        activation: np.ndarray [shape=(T, 360)]
            The raw activation matrix
    """
    activation = get_activation(audio, sr, model_capacity=model_capacity,
                                center=center, step_size=step_size,
                                verbose=verbose)
    confidence = activation.max(axis=1)

    if viterbi:
        cents = to_viterbi_cents(activation)
    else:
        cents = to_local_average_cents(activation)

    frequency = 10 * 2 ** (cents / 1200)
    frequency[np.isnan(frequency)] = 0

    time = np.arange(confidence.shape[0]) * step_size / 1000.0

    # z = np.reshape(activation, [-1, 6, 60])
    # z = np.mean(z, axis=1) #(None, 60)
    # z = np.reshape(z, [-1,12,5])
    # z = np.mean(z, axis=2)  # (None, 12)
    zarg = np.argmax(activation, axis=1)
    zarg = zarg%60
    zarg = zarg / 5
    # ((((cents - 1997.3794084376191) / 20) % 60) / 5)
    return time, frequency, zarg, confidence, activation


def process_file(file, output=None, model_capacity='full', viterbi=False,
                 center=True, save_activation=False, save_plot=False,
                 plot_voicing=False, step_size=10, verbose=True):
    """
    Use the input model to perform pitch estimation on the input file.

    Parameters
    ----------
    file : str
        Path to WAV file to be analyzed.
    output : str or None
        Path to directory for saving output files. If None, output files will
        be saved to the directory containing the input file.
    model_capacity : 'tiny', 'small', 'medium', 'large', or 'full'
        String specifying the model capacity; see the docstring of
        :func:`~crepe.core.build_and_load_model`
    viterbi : bool
        Apply viterbi smoothing to the estimated pitch curve. False by default.
    center : boolean
        - If `True` (default), the signal `audio` is padded so that frame
          `D[:, t]` is centered at `audio[t * hop_length]`.
        - If `False`, then `D[:, t]` begins at `audio[t * hop_length]`
    save_activation : bool
        Save the output activation matrix to an .npy file. False by default.
    save_plot : bool
        Save a plot of the output activation matrix to a .png file. False by
        default.
    plot_voicing : bool
        Include a visual representation of the voicing activity detection in
        the plot of the output activation matrix. False by default, only
        relevant if save_plot is True.
    step_size : int
        The step size in milliseconds for running pitch estimation.
    verbose : bool
        Print status messages and keras progress (default=True).

    Returns
    -------

    """
    try:
        sr, audio = wavfile.read(file)
    except ValueError:
        print("CREPE: Could not read %s" % file, file=sys.stderr)
        raise

    time, frequency,cents, confidence, activation = predict(
        audio, sr,
        model_capacity=model_capacity,
        viterbi=viterbi,
        center=center,
        step_size=step_size,
        verbose=1 * verbose)

    # write prediction as TSV
    f0_file = output_path(file, ".f0.csv", output)
    f0_data = np.vstack([time, frequency, cents, confidence]).transpose()
    np.savetxt(f0_file, f0_data, fmt=['%.3f', '%.3f', '%.6f', '%.6f'], delimiter=',',
               header='time,frequency,cents,confidence', comments='')
    if verbose:
        print("CREPE: Saved the estimated frequencies and confidence values "
              "at {}".format(f0_file))

    # save the salience file to a .npy file
    if save_activation:
        activation_path = output_path(file, ".activation.npy", output)
        np.save(activation_path, activation)
        if verbose:
            print("CREPE: Saved the activation matrix at {}".format(
                activation_path))

    # save the salience visualization in a PNG file
    if save_plot:
        import matplotlib.cm
        from imageio import imwrite

        plot_file = output_path(file, ".activation.png", output)
        # to draw the low pitches in the bottom
        salience = np.flip(activation, axis=1)
        inferno = matplotlib.cm.get_cmap('inferno')
        image = inferno(salience.transpose())

        if plot_voicing:
            # attach a soft and hard voicing detection result under the
            # salience plot
            image = np.pad(image, [(0, 20), (0, 0), (0, 0)], mode='constant')
            image[-20:-10, :, :] = inferno(confidence)[np.newaxis, :, :]
            image[-10:, :, :] = (
                inferno((confidence > 0.5).astype(np.float))[np.newaxis, :, :])

        imwrite(plot_file, (255 * image).astype(np.uint8))
        if verbose:
            print("CREPE: Saved the salience plot at {}".format(plot_file))

