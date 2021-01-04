from __future__ import division
from __future__ import print_function

import os
# import re
import sys
import math
from data_generator import DataGenerator
from scipy.io import wavfile
import numpy as np
import h5py
from numpy.lib.stride_tricks import as_strided
import tensorflow as tf
import tensorflow_transform as tft
# tf.config.run_functions_eagerly(True)
# tf.compat.v1.disable_eager_execution()
from tensorflow.keras.layers import Input, Reshape, Conv2D, BatchNormalization, Softmax, Conv1D, Bidirectional
from tensorflow.keras.layers import MaxPool2D, Dropout, Permute, Flatten, Dense, MaxPool1D, AvgPool1D, Bidirectional, \
    LSTM, Lambda
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
import librosa
import librosa.display
from encoder_2 import Encoder
import pyhocon
import os
import json
import pandas as pd
from resnet import ResnetBuilder
from collections import defaultdict
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

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

res_net = ResNet50(include_top=False, input_shape=(60, 60, 3))
# for layer in res_net.layers[:143]:
#     layer.trainable = False
# for layer in res_net.layers[143:]:
#     layer.trainable = True
for layer in res_net.layers:
    layer.trainable = False


# model_2 = ResNet50(include_top=False, input_shape=(60, 60, 3))
# for layer in model_2.layers:
#     layer.trainable = False

# the model is trained on 16kHz audio
# model_srate = 16000
# max_batch_size = 3000
# sequence_length = 200
# n_labels = 30
# config = pyhocon.ConfigFactory.parse_file("crepe/experiments.conf")['test']

def build_and_load_model(config, task='raga'):
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
    hop_size = int(config['hop_size'] * model_srate)
    sequence_length = int((config['sequence_length'] * model_srate - 1024) / hop_size) + 1
    drop_rate = config['drop_rate']
    drop_rate_tonic = config['drop_rate_tonic']
    drop_rate_raga = config['drop_rate_raga']
    cutoff = config['cutoff']
    n_frames = 1 + int((model_srate * cutoff - 1024) / hop_size)
    n_seq = int(n_frames // sequence_length)

    n_labels = config['n_labels']

    note_dim = config['note_dim']

    # x = Input(shape=(1024,), name='input2', dtype='float32')
    x_batch = Input(shape=(None, 1024), name='x_input', dtype='float32')
    tonic_batch = Input(shape=(60,), name='tonic_input', dtype='float32')
    pitches_batch = Input(shape=(None, 360), name='pitches_input', dtype='float32')
    random_batch = Input(shape=(), name='random_input', dtype='int32')
    cqt_batch = Input(shape=(60,), name='cqt_input', dtype='float32')
    # transpose_by_batch = Input(shape=(), name='transpose_input', dtype='int32')

    x = x_batch[0]
    tonic_input = tf.expand_dims(tonic_batch[0], 0)
    pitches = pitches_batch[0]
    random = random_batch[0]
    cqt = cqt_batch[0]
    # transpose_by = transpose_by_batch[0]

    transpose_by = tf.random.uniform(shape=(), minval=0, maxval=60, dtype=tf.int32)
    transpose_by = transpose_by * random
    random = tf.cast(random, tf.float32)

    tonic_input = tf.roll(tonic_input, -transpose_by, axis=-1)
    y, note_emb = get_pitch_emb(x, n_seq, n_frames, model_capacity)
    pitch_model = Model(inputs=[x_batch], outputs=y)

    if task == 'pitch':
        return pitch_model
    # pitch_model.load_weights('model/hindustani_pitch_model.hdf5', by_name=True)
    pitch_model.load_weights('model/model-full.h5', by_name=True)

    note_emb = reduce_note_emb_dimensions(note_emb, note_dim)

    red_y = tf.reshape(pitches, [-1, 6, 60])
    red_y = tf.reduce_sum(red_y, axis=1)  # (None, 60)
    red_y = tf.roll(red_y, -transpose_by, axis=1)
    cqt = tf.roll(cqt, -transpose_by, axis=-1)
    # red_y_random = apply_note_random(red_y)
    # red_y = random*red_y_random + (1-random)*red_y
    # Tonic
    tonic_logits = get_tonic_emb(red_y, cqt, note_emb, note_dim, drop_rate_tonic)
    # tonic_emb = get_tonic_from_cqt(cqt, drop_rate=drop_rate_tonic)

    # tonic_logits = Dense(60, activation='sigmoid')(tonic_emb)

    # tonic_model = Model(inputs=[pitches_batch, random_batch], outputs=tonic_logits)
    # tonic_model.summary()
    # tonic_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    #
    # tonic_model.load_weights('model/hindustani_raga_model.hdf5', by_name=True)

    # if task== 'tonic':
    #     return tonic_model

    # for layer in tonic_model.layers:
    #     layer.trainable = False

    # tonic_logits_masked = tonic_logits[0]
    # tonic_logits_pad = tf.pad(tonic_logits_masked, [[5,5]])
    # tonic_logits_argmax = tf.cast(tf.argmax(tonic_logits_pad), tf.int32)
    # tonic_indices = tf.range(70)
    # lower_limit = tf.less(tonic_indices, tonic_logits_argmax-4)
    # upper_limit = tf.greater(tonic_indices, tonic_logits_argmax + 5)
    # tonic_logits_mask = 1 - tf.cast(tf.logical_or(lower_limit, upper_limit), tf.float32)
    # tonic_logits_mask = tonic_logits_mask[5:-5]
    # tonic_logits_masked = tf.multiply(tonic_logits_masked, tonic_logits_mask)
    # tonic_logits_masked = tonic_logits_masked/tf.reduce_sum(tonic_logits_masked)
    # tonic_logits_masked = tf.expand_dims(tonic_logits_masked, 0)

    # rag_emb = get_raga_emb(red_y, tonic_logits, note_emb, note_dim, drop_rate_raga)
    raga_logits = get_raga_emb(red_y, cqt, tonic_logits, note_emb, note_dim, drop_rate_raga)

    # tonic_emb = ffnn(tf.concat([rag_emb, tonic_emb], axis=1), [4*note_dim, 2*note_dim, 2*note_dim], drop_rate_raga)
    # tonic_logits = Dense(60, activation='sigmoid')(tonic_emb)

    # raga_logits = Dense(n_labels, activation='softmax', name='raga')(rag_emb)

    loss_weights = config['loss_weights']

    # rag_model = Model(inputs=[pitches_batch, tonic_batch], outputs=[raga_logits])
    # rag_model = Model(inputs=[pitches_batch, transpose_by_batch], outputs=[tonic_logits, raga_logits])
    # rag_model = Model(inputs=[pitches_batch, transpose_by_batch], outputs=[tonic_logits, raga_logits])
    # rag_model = Model(inputs=[pitches_batch, transpose_by_batch], outputs=[tonic_logits, raga_logits])
    tonic_logits_roll = tf.roll(tonic_logits, transpose_by, axis=-1, name='tonic')

    rag_model = Model(inputs=[pitches_batch, random_batch, cqt_batch], outputs=[raga_logits, tonic_logits_roll])
    # rag_model = Model(inputs=[pitches_batch, random_batch], outputs=[raga_logits])
    # rag_model = Model(inputs=[pitches_batch, tonic_batch], outputs=[raga_logits])
    # rag_model = Model(inputs=[x_batch], outputs=[tonic_logits, raga_logits])
    # rag_model.compile(loss={'tonic': 'binary_crossentropy', 'raga': 'categorical_crossentropy'},
    #               optimizer='adam', metrics={'tonic': 'categorical_accuracy', 'raga': 'accuracy'}, loss_weights={'tonic': loss_weights[0], 'raga': loss_weights[1]})
    cce = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.0)
    rag_model.compile(loss={'raga': cce, 'tf_op_layer_tonic': 'binary_crossentropy'},
                      optimizer='adam', metrics={'raga': 'accuracy', 'tf_op_layer_tonic': 'accuracy'},
                      loss_weights={'raga': loss_weights[0], 'tf_op_layer_tonic': loss_weights[1]})
    rag_model.load_weights('model/hindustani_raga_model.hdf5', by_name=True)

    # rag_model.compile(loss={'raga': 'categorical_crossentropy'},
    #                   optimizer='adam', metrics={'raga': 'accuracy'},
    #                   loss_weights={'raga': loss_weights[0]})

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
    for i in range(n_seq):
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


def get_top_notes(red_y):
    c_note = freq_to_cents(31.7 * 2, 25)
    c_note = np.reshape(c_note, [6, 60])
    c_note = np.sum(c_note, axis=0)

    diag_tf = tf.math.reduce_std(red_y, axis=0)
    # diag_tf = tf.reduce_mean(red_y, axis=0)
    diag_tf = AvgPool1D(pool_size=2, strides=1, padding='same')(tf.expand_dims(tf.expand_dims(diag_tf, 0), 2))[0, :, 0]
    diag_tf_p = tf.roll(diag_tf, 1, 0)
    diag_tf_n = tf.roll(diag_tf, -1, 0)
    diag_tf_1 = tf.less_equal(diag_tf_p, diag_tf)
    diag_tf_2 = tf.less_equal(diag_tf_n, diag_tf)
    diag_tf_3 = tf.logical_and(diag_tf_1, diag_tf_2)
    diag_tf_4 = tf.multiply(diag_tf, tf.cast(diag_tf_3, tf.float32))
    diag_tf_4 = tf.cast(diag_tf_4, tf.float32)
    diag_tf_3_ind = tf.where(diag_tf_3)[:, 0]
    diag_tf_3_soft = tf.keras.layers.Lambda(
        lambda x: tf.map_fn(lambda a: tf.roll(tf.constant(c_note), a, axis=-1), x, tf.float64))(diag_tf_3_ind)
    diag_tf_3_soft = tf.reduce_sum(diag_tf_3_soft, axis=0)
    diag_tf_3_soft = tf.cast(diag_tf_3_soft, tf.float32)

    diag_tf_3 = tf.cast(diag_tf_3, tf.float32)
    return diag_tf_4, diag_tf_3_soft, diag_tf_3
    # return diag_tf_3, diag_tf_3


def get_peak_notes(hist):
    diag_tf_p = tf.roll(hist, 1, 0)
    diag_tf_n = tf.roll(hist, -1, 0)
    diag_tf_1 = tf.less_equal(diag_tf_p, hist)
    diag_tf_2 = tf.less_equal(diag_tf_n, hist)
    diag_tf_3 = tf.logical_and(diag_tf_1, diag_tf_2)
    diag_tf_4 = tf.multiply(hist, tf.cast(diag_tf_3, tf.float32))

    return diag_tf_4


def get_hist_emb(red_y, cqt, note_emb, note_dim, indices, topk, drop_rate=0.2):
    top_notes, top_notes_soft, top_notes_hard = get_top_notes(red_y)
    emb_dot = tf.reduce_mean(tf.multiply(note_emb, tf.tile(note_emb[0:1], [60, 1])), axis=1)
    is_tonic = indices is None
    emb_dot = min_max_scale(emb_dot, is_tonic)
    # emb_dot = tf.tile(tf.expand_dims(emb_dot,0), [topk,1])[0]
    hist = tf.reduce_mean(red_y, axis=0)

    hist = min_max_scale(hist, is_tonic)
    # top_notes_soft = min_max_scale(top_notes_soft+top_notes)
    top_notes_soft = min_max_scale(top_notes_soft, is_tonic)
    cqt_mean = min_max_scale(cqt, is_tonic)
    # cqt_std = min_max_scale(tf.math.reduce_std(cqt, axis=1))
    top_notes = min_max_scale(top_notes, is_tonic)

    # hist_cc = tf.expand_dims(hist,1)
    hist_cc_all = []
    # cqt = min_max_scale(tf.reduce_mean(cqt,axis=1))

    hist_cc = tf.transpose(tf.stack([cqt_mean, hist, top_notes, top_notes_soft]))
    for i in range(topk):
        hist_cc_trans = tf.roll(hist_cc, -indices[i], axis=0)
        hist_cc_all.append(hist_cc_trans)
    # values = tf.cast(values, tf.float32)
    hist_cc_all = tf.stack(hist_cc_all)

    # if indices is None:
    #     hist_cc = tf.transpose(tf.stack([cqt_mean, hist, top_notes, top_notes_soft]))
    #
    #     # hist_cc = tf.transpose(tf.stack([hist, top_notes]))
    #     hist_cc_trans = tf.roll(hist_cc, 0, axis=0)
    #     hist_cc_all.append(hist_cc_trans)
    #     hist_cc_all = tf.stack(hist_cc_all)
    # else:
    #     hist_cc = tf.transpose(tf.stack([cqt_mean, hist, top_notes, top_notes_soft]))
    #     for i in range(topk):
    #         hist_cc_trans = tf.roll(hist_cc, -indices[i], axis=0)
    #         hist_cc_all.append(hist_cc_trans)
    #     # values = tf.cast(values, tf.float32)
    #     hist_cc_all = tf.stack(hist_cc_all)
        # values = tf.expand_dims(tf.expand_dims(values,1),2)
        # hist_cc_all = tf.reduce_sum(tf.multiply(hist_cc_all, values), axis=0, keepdims=True)
        # hist_cc_all = hist_cc_all/tf.reduce_sum(values)

    # res_net = ResNet50(include_top=False, input_shape=(60, 60, 3))

    # for layer in res_net.layers[:143]:
    #     layer.trainable = False
    # for layer in res_net.layers[143:]:
    #     layer.trainable = True
    #
    # for layer in res_net.layers:
    #     layer.trainable = False
    # for layer in res_net.layers:
    #     if "BatchNormalization" in layer.__class__.__name__:
    #         layer.trainable = True

    # model = Model(inputs=res_net.input, outputs=res_net.output)
    # model = res_net

    # hist_cc_all = tf.expand_dims(hist_cc_all,3)
    # hist_cc_all = tf.transpose(hist_cc_all, [0, 2, 1,3])
    # matmul = tf.matmul(hist_cc_all, hist_cc_all, transpose_b=True)
    # matmul = tf.transpose(matmul, [0, 2, 3, 1])
    #
    # matmul = tf.keras.applications.resnet.preprocess_input(matmul)
    # z = model(matmul)
    # z = Flatten()(z)
    # z = Dropout(drop_rate)(z)
    # z = Dense(2 * note_dim, activation='relu')(z)
    # return z, top_notes

    # if indices is None:
    #
    #     kernel_size = [5, 10, 15, 20]
    #     tonic_cnn_filters = 768
    #     outputs = []
    #     for ks in kernel_size:
    #         bz = tf.concat([hist_cc_all, hist_cc_all[:,:ks,:]], axis=1)
    #         # bz = tf.reshape(bz, [1, -1, 1]) #
    #         conv = Conv1D(filters=tonic_cnn_filters, kernel_size=ks, strides=1, activation='relu', padding='valid')(bz) ##((60+ks)/ks, ks, 1)
    #         conv = tf.squeeze(conv, axis=0)
    #         conv = conv[:-1, :]
    #         conv = Dropout(drop_rate)(conv)
    #         conv = Dense(2*note_dim, activation='relu')(conv)
    #         outputs.append(conv)
    #
    #     outputs = tf.concat(outputs, 1)
    #     outputs = ffnn(outputs, [2*note_dim], drop_rate=drop_rate)
    #     return outputs, top_notes

    # hist_cc_all = tf.concat([hist_cc_all, tf.expand_dims(emb_dot,2)], axis=2)

    z = Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')(hist_cc_all)
    z = Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')(z)
    z = BatchNormalization()(z)
    z = MaxPool1D(pool_size=2)(z)
    z = Dropout(drop_rate)(z)
    z = Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(z)
    z = Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(z)
    z = BatchNormalization()(z)
    z = MaxPool1D(pool_size=2)(z)
    z = Dropout(drop_rate)(z)
    z = Conv1D(filters=192, kernel_size=3, strides=1, padding='same', activation='relu')(z)
    z = Conv1D(filters=192, kernel_size=3, strides=1, padding='same', activation='relu')(z)
    z = BatchNormalization()(z)
    z = MaxPool1D(pool_size=2)(z)
    z = Dropout(drop_rate)(z)
    z = Flatten()(z)
    # z = tf.concat([z, diag_tf_3_den], axis=1)
    z = Dense(2 * note_dim, activation='relu')(z)
    z = Dropout(drop_rate)(z)
    return z, top_notes_hard


def get_tonic_emb(red_y, cqt, note_emb, note_dim, drop_rate=0.2):
    topk=5
    indices = tf.random.uniform(shape=(5,), minval=0, maxval=60, dtype=tf.int32)
    hist_emb, top_notes = get_hist_emb(red_y, cqt, note_emb, note_dim, indices, topk, drop_rate)
    # ndms = get_ndms(red_y, top_notes, None, 1, note_emb, note_dim, drop_rate)
    # tonic_emb = combine(hist_emb, ndms, note_dim, drop_rate)

    tonic_emb = hist_emb

    tonic_logits_norm = []
    tonic_logits = Dense(60, activation='sigmoid')(tonic_emb)
    for i in range(topk):
        tonic_logits_norm.append(tf.roll(tonic_logits[i],indices[i], axis=0))
    tonic_logits = tf.reduce_mean(tonic_logits_norm, axis=0, keepdims=True)

    return tonic_logits


def get_raga_emb(red_y, cqt, tonic_logits, note_emb, note_dim, drop_rate=0.2):
    # rs = -5
    # re = 5
    topk = 5
    # indices = tf.range(topk)

    # tonic_logits = Dense(1, activation='sigmoid')(tonic_emb)
    # tonic_logits = tf.transpose(tonic_logits)
    # peak_notes = get_peak_notes(tonic_logits[0])

    tonic_argmax = tf.argmax(tonic_logits[0])
    indices = tf.range(tonic_argmax - 2, tonic_argmax + 3)
    indices = tf.math.mod(60 + indices, 60)
    values = tf.gather(tonic_logits[0], indices)

    # tonic_logits = Dense(1, activation='sigmoid')(tonic_emb)
    #
    # peak_notes = get_peak_notes(tonic_logits[0])
    # values, indices = tf.nn.top_k(peak_notes,topk)
    # tonic_logits = tf.transpose(tonic_logits)
    # concat = tf.concat([raga_emb, tonic_emb], axis=-1)
    # concat = ffnn(concat, [2*note_dim, note_dim], drop_rate)
    # tonic_logits = Dense(1, activation='sigmoid')(concat)
    # tonic_logits_argmax = tf.argmax(tonic_logits, axis=0)[0]
    # indices = tf.range(-4+tonic_logits_argmax, tonic_logits_argmax+5)
    # indices = tf.math.mod(60+indices, 60)

    # tonic_emb = tf.tile(tonic_emb, [60,1])
    rs = 0
    re = 59
    # rs = topk//2
    # re = topk//2+1

    # tonic_argmax = tf.argmax(tonic_logits[0])
    # tonic_concat = tf.concat([tonic_logits[0], tonic_logits[0], tonic_logits[0]], axis=-1)
    # values = tonic_concat[60+tonic_argmax-rs:60+tonic_argmax+re]

    # c_note = freq_to_cents(31.7*2, 25)
    # c_note = np.reshape(c_note, [6, 60])
    # c_note = np.sum(c_note, axis=0)
    # values = tf.roll(c_note, tonic_argmax, axis=-1)
    # values = tf.cast(values, tf.float32)
    # values = tf.concat([values, values, values], axis=-1)
    # values = values[60+tonic_argmax-rs:60+tonic_argmax+re]

    # indices = tf.range(tonic_argmax-rs, tonic_argmax+re)
    # indices = tf.math.mod(60+indices, 60)

    # values, indices = conv_tonic(tonic_logits, topk=topk)
    # values = tf.cast(values, tf.float32)
    # values = tf.cast(tonic_logits[0], tf.float64)
    # indices = tf.range(60)
    # values, indices = tf.nn.top_k(tonic_logits[0], k=topk)

    # transpose_by = tf.cast(tf.argmax(tonic_logits, axis=1)[0], tf.int32)
    # red_y = tf.roll(red_y, -transpose_by, axis=1)
    hist_emb, top_notes = get_hist_emb(red_y, cqt, note_emb, note_dim, indices, topk, drop_rate)
    red_y_clustered = cluster_top_notes(red_y, top_notes)
    ndms, red_y_am = get_ndms(red_y_clustered, top_notes, indices, topk, note_emb, note_dim, drop_rate)
    # rnn = get_rag_from_rnn(red_y_am, indices, topk, note_emb, note_dim, drop_rate)
    raga_emb = combine(hist_emb, ndms, note_dim, drop_rate)
    # raga_emb = combine(raga_emb, rnn, note_dim, drop_rate)
    # raga_emb = ndms
    # raga_emb = hist_emb
    # concat = tf.concat([raga_emb, tf.tile(tonic_emb, [60,1])], axis=1)
    # concat = ffnn(concat, [2*note_dim,1], drop_rate)
    # concat = tf.transpose(concat)
    # tonic_logits = Dense(60, activation='sigmoid')(concat)
    # tonic_logits = tf.expand_dims(tf.nn.softmax(tf.transpose(concat)[0]),1)
    # tonic_logits = tf.transpose(tonic_logits)
    # peak_notes = get_peak_notes(tonic_logits[0])

    raga_emb = tf.reduce_sum(tf.multiply(raga_emb, tf.expand_dims(values, 1)), axis=0, keepdims=True) / tf.reduce_sum(
        values)

    # raga_emb = tf.reduce_sum(tf.multiply(raga_emb, tf.transpose(tonic_logits)), axis=0, keepdims=True)

    # raga_emb = tf.gather(raga_emb, indices)
    # tonic_logits_gather = tf.gather(tonic_logits, indices)
    # raga_emb = tf.reduce_sum(tf.multiply(raga_emb, tonic_logits_gather), axis=0, keepdims=True)/tf.reduce_sum(tonic_logits_gather)
    # raga_emb = tf.reduce_sum(tf.multiply(raga_emb, tonic_logits), axis=0, keepdims=True)/tf.reduce_sum(tonic_logits)

    # raga_emb = tf.reduce_sum(tf.multiply(raga_emb, tf.expand_dims(values,1)), axis=0, keepdims=True)/tf.reduce_sum(values)
    raga_logits = Dense(30, activation='softmax', name='raga')(raga_emb)
    # raga_logits = tf.nn.softmax(raga_logits, axis=-1, name='raga')
    # raga_logits_argmax = tf.argmax(tf.multiply(tf.reduce_max(raga_logits,1), values))
    # raga_logits = tf.expand_dims(raga_logits[raga_logits_argmax],0, name='raga')
    # raga_emb = tf.multiply(raga_emb, tf.expand_dims(values,1))
    # raga_emb = tf.reduce_sum(raga_emb, axis=0, keepdims=True)
    # raga_emb = raga_emb/tf.reduce_sum(values)

    return raga_logits


def apply_note_random(red_y):
    def apply_random_roll(a):
        r = tf.random.uniform(shape=(), minval=-2, maxval=3, dtype=tf.int32)
        return tf.roll(a, r, axis=-1)

    red_y_roll = tf.keras.layers.Lambda(apply_random_roll)(red_y)
    return red_y_roll


def combine(emb1, emb2, note_dim, drop_rate):
    emb = Dense(2 * note_dim, activation='relu')(tf.concat([emb1, emb2], axis=1))
    emb = Dropout(drop_rate)(emb)
    f = tf.nn.sigmoid(Dense(2 * note_dim)(emb))
    emb = f * emb1 + (1 - f) * emb2
    return emb


def freq_to_cents(freq, std=25):
    frequency_reference = 10
    c_true = 1200 * math.log(freq / frequency_reference, 2)

    cents_mapping = np.linspace(0, 7180, 360) + 1997.3794084376191
    target = np.exp(-(cents_mapping - c_true) ** 2 / (2 * std ** 2))
    return target


def one_hot_note_emb():
    cents_mapping = tf.range(0, 60)
    target = tf.math.exp(-(cents_mapping - 29) ** 2 / (2 * 2 ** 2))
    target = tf.roll(target, -29, axis=-1)
    note_embs = []
    for i in range(60):
        note_embs.append(tf.roll(target, i, axis=-1))
    note_embs = tf.stack(note_embs)
    return note_embs


def get_ndms(red_y_am, top_notes, indices, topk, note_emb, note_dim, drop_rate=0.6):
    is_tonic = indices is None
    note_emb_mat = tf.matmul(note_emb, note_emb, transpose_b=True)
    note_emb_mat = note_emb_mat / note_dim
    note_emb_mat = tf.cast(note_emb_mat, tf.float64)
    top_notes_tile = tf.tile(tf.expand_dims(top_notes, 0), [tf.shape(red_y_am)[0], 1])
    # red_y_am_base = tf.argmax(red_y, axis=1)
    # red_y_am = tf.one_hot(red_y_am_base, 60)
    # red_y_am = tf.multiply(top_notes_tile, red_y_am)
    # red_y_am_nz = tf.reduce_sum(red_y_am, axis=1)
    # red_y_am_nz = tf.where(red_y_am_nz)[:, 0]
    # red_y_am = tf.gather(red_y_am, red_y_am_nz)
    # red_y_am = tf.argmax(red_y_am, 1)

    red_y_am = get_unique_seq_1(red_y_am)
    red_y_am_first = red_y_am
    matmul_1 = get_ndms_mat(red_y_am)

    red_y_am = get_unique_seq_2(red_y_am)
    red_y_am = get_unique_seq_1(red_y_am)
    matmul_2 = get_ndms_mat(red_y_am)

    red_y_am = get_unique_seq_2(red_y_am)
    red_y_am = get_unique_seq_1(red_y_am)
    matmul_3 = get_ndms_mat(red_y_am)

    matmul_1 = min_max_scale(matmul_1, is_tonic)
    matmul_2 = min_max_scale(matmul_2, is_tonic)
    matmul_3 = min_max_scale(matmul_3, is_tonic)
    # matmul_3 = matmul_3/tf.cast(tf.shape(red_y_am)[0], tf.float32)
    # matmul_4 = matmul_4 / tf.cast(tf.shape(red_y_am)[0], tf.float32)
    matmul = tf.stack([matmul_1, matmul_2, matmul_3, note_emb_mat], axis=2)
    # matmul = tf.stack([matmul_1], axis=2)
    # matmul = tf.stack([matmul_1, matmul_3], axis=2)
    ndms = []

    if indices is None:
        matmul_tmp = tf.roll(matmul, 0, axis=0)
        matmul_tmp = tf.roll(matmul_tmp, 0, axis=1)
        ndms.append(matmul_tmp)
        ndms = tf.stack(ndms)
    else:
        for i in range(topk):
            matmul_tmp = tf.roll(matmul, [-indices[i], -indices[i]], axis=[0, 1])
            ndms.append(matmul_tmp)
        ndms = tf.stack(ndms)
        # values = tf.expand_dims(tf.expand_dims(tf.expand_dims(values,1),2),3)
        # ndms = tf.reduce_sum(tf.multiply(ndms, values), axis=0, keepdims=True)
        # ndms = ndms/tf.reduce_sum(values)
    # res_net = ResNet50(include_top=False, input_shape=(60, 60, 3))
    # for layer in res_net.layers[:143]:
    #     layer.trainable = False
    # for layer in res_net.layers[143:]:
    #     layer.trainable = True
    # for layer in res_net.layers:
    #     layer.trainable = True
    #
    # for layer in res_net.layers:
    #     if "BatchNormalization" in layer.__class__.__name__:
    #         layer.trainable = True

    # model = Model(inputs=res_net.input, outputs=res_net.output)
    # model = res_net
    # ndms = ndms*255
    # ndms = tf.keras.applications.resnet.preprocess_input(ndms)
    # z = model(ndms)
    # z = Flatten()(z)
    # z = Dense(2 * note_dim, activation='relu')(z)
    # z = Dropout(drop_rate)(z)
    # z = model.predict(ndms)
    # z = Flatten()(z)
    # z = Dense(2 * note_dim, activation='relu')(z)
    # return z

    # ndms = tf.concat([ndms, tf.expand_dims(note_emb_mat,3)], axis=3)
    # ndms = BatchNormalization()(ndms)
    # resnet = ResnetBuilder.build_resnet_18((60,60,3), None)
    # z = resnet(ndms)
    # # z = Dropout(drop_rate)(z)
    # z = Dense(2*note_dim)(z)
    # return z
    # z = Conv2D(kernel_size=(5,5), filters=64, activation='relu', padding='valid')(ndms)
    # z = BatchNormalization()(z)
    # z = MaxPool2D(pool_size=(2,2), strides=None, padding='valid')(z)
    # z = Dropout(drop_rate)(z)
    # z = Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='valid')(z)
    # z = BatchNormalization()(z)
    # z = MaxPool2D(pool_size=(2,2), strides=None, padding='valid')(z)
    # z = Dropout(drop_rate)(z)
    # z = Flatten()(z)
    # z = Dense(2 * note_dim, activation='relu')(z)
    # z = Dropout(drop_rate)(z)

    z = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(ndms)
    z = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(z)
    z = BatchNormalization()(z)
    z = MaxPool2D(pool_size=(2, 2), strides=None, padding='valid')(z)
    z = Dropout(drop_rate)(z)
    z = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(z)
    z = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(z)
    z = BatchNormalization()(z)
    z = MaxPool2D(pool_size=(2, 2), strides=None, padding='valid')(z)
    z = Dropout(drop_rate)(z)
    z = Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(z)
    z = Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(z)
    z = BatchNormalization()(z)
    z = MaxPool2D(pool_size=(2, 2), strides=None, padding='valid')(z)
    # z = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(z)
    # z = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(z)
    # z = BatchNormalization()(z)
    # z = MaxPool2D(pool_size=(2, 2), strides=None, padding='valid')(z)
    # z = Dropout(drop_rate)(z)
    z = Flatten()(z)
    z = Dense(2 * note_dim, activation='relu')(z)
    z = Dropout(drop_rate)(z)
    return z, red_y_am_first


def cluster_top_notes(red_y, top_notes):
    c_note = freq_to_cents(31.7 * 2)
    c_note = np.reshape(c_note, [6, 60])
    c_note = np.sum(c_note, axis=0)

    top_notes_where = tf.where(top_notes)[:, 0]
    top_notes_where = tf.tile(tf.expand_dims(top_notes_where, 0), [tf.shape(red_y)[0], 1])
    red_y_arg = tf.argmax(red_y, axis=1)
    red_y_arg = tf.expand_dims(red_y_arg, 1)
    diff = tf.math.abs(red_y_arg - top_notes_where)
    diff = tf.argmin(diff, 1)
    diff_ohe = tf.one_hot(diff, tf.shape(top_notes_where)[1])
    top_notes_where = tf.multiply(tf.cast(top_notes_where, tf.float32), diff_ohe)
    top_notes_where = tf.reduce_max(top_notes_where, axis=1)
    top_notes_where = tf.cast(top_notes_where, tf.int32)
    # top_notes_where = tf.keras.layers.Lambda(
    #     lambda x: tf.map_fn(lambda a: tf.roll(tf.constant(c_note), a, axis=-1), x, tf.float64))(top_notes_where)
    return top_notes_where


def get_unique_seq_1(arg_y):
    # red_y = tf.random.uniform(shape=(100,), maxval=60, dtype=tf.int32)
    # red_y  = tf.one_hot(red_y,60)

    arg_y = tf.concat([[0.], tf.cast(arg_y, tf.float32)], axis=-1)  # None+1

    arg_y_shifted = tf.roll(arg_y, -1, axis=-1)  # 1,None+1

    mask = tf.cast(tf.not_equal(arg_y, arg_y_shifted), tf.float32)  # 1,None+1
    mask = tf.where(mask)[:, 0]
    uni_seq_notes = tf.gather(arg_y_shifted, mask)
    uni_seq_notes = tf.cast(uni_seq_notes, tf.int32)
    return uni_seq_notes


def get_unique_seq_2(ndms):
    temp = tf.equal(ndms, tf.roll(ndms, -2, axis=-1))
    temp1 = tf.cast(tf.logical_not(tf.roll(temp, 1, axis=-1)), tf.int32)
    temp2 = tf.cast(temp, tf.int32)

    uni_seq_notes = tf.multiply(ndms, temp1) + tf.roll(tf.multiply(ndms, temp2), 1, axis=-1)
    return uni_seq_notes


def conv_tonic(pred_tonic, topk):
    pred_tonic = pred_tonic[0]
    kl = tf.keras.losses.BinaryCrossentropy()
    c_note = freq_to_cents(31.7 * 2, 25)
    c_note = np.reshape(c_note, [6, 60])
    c_note = np.sum(c_note, axis=0)
    bces = []
    for i in range(60):
        bces.append(kl(pred_tonic, np.roll(c_note, i, axis=-1)))
    bces = tf.stack(bces)
    bces_argmin = tf.argmin(bces)

    pred_tonic = tf.concat([pred_tonic, pred_tonic, pred_tonic], axis=-1)
    # bces_1 = tf.concat([c_note, c_note, c_note], axis=-1)
    rs = topk // 2
    re = topk // 2 + 1
    values = pred_tonic[60 + bces_argmin - rs:60 + bces_argmin + re]

    # c_note = tf.roll(c_note,bces_argmin, axis=-1)
    # c_note = tf.concat([c_note, c_note, c_note], axis=-1)
    # values = c_note[60+bces_argmin-rs:60+bces_argmin+re]
    # values_1 = bces_1[60 + bces_argmin - 5:60 + bces_argmin + 5]
    # values = tf.multiply(values, values_1)
    # values = tf.roll(c_note, bces_argmin, axis=-1)
    indices = tf.range(bces_argmin - rs, bces_argmin + re)
    indices = tf.math.mod(60 + indices, 60)
    return values, indices


def get_ndms_mat(ndms):
    c_note = freq_to_cents(65.41, 25)
    c_note = np.reshape(c_note, [6, 60])
    c_note = np.sum(c_note, axis=0)
    ndms_ohe = tf.keras.layers.Lambda(
        lambda x: tf.map_fn(lambda a: tf.roll(tf.constant(c_note), a, axis=-1), x, tf.float64))(ndms)
    ndms_roll_ohe = tf.roll(ndms_ohe, -1, axis=0)
    # ndms_roll_ohe = tf.keras.layers.Lambda(lambda x: tf.map_fn(lambda a: tf.roll(tf.constant(c_note), a, axis=-1), x, tf.float64))(ndms_roll)
    # ndms_roll_ohe = tf.roll(ndms_ohe, -1, axis=0)
    # ndms_ohe = tf.one_hot(ndms, 60)
    # ndms_roll = tf.roll(ndms, -1, axis=-1)
    # ndms_roll_ohe = tf.one_hot(ndms_roll, 60)
    matmul = tf.matmul(ndms_ohe, ndms_roll_ohe, transpose_a=True)

    return matmul


def get_tonic_from_cqt(cqt, tonic_emb_size=128, tonic_cnn_filters=128, drop_rate=0.1):
    # chroma, int(note_dim*3/32), int(note_dim/2), drop_rate
    chroma = tf.expand_dims(cqt, 0)
    chroma = tf.expand_dims(chroma, 3)
    y = Conv2D(tonic_cnn_filters, (5, 5), strides=1, padding='same', activation='relu', name='tonic_cnn_1')(chroma)
    y = Conv2D(tonic_cnn_filters, (5, 5), strides=1, padding='same', activation='relu', name='tonic_cnn_2')(y)
    y = Conv2D(tonic_cnn_filters, (5, 5), strides=1, padding='same', activation='relu', name='tonic_cnn_3')(y)
    y = Dropout(drop_rate)(y)
    y = Conv2D(tonic_cnn_filters, (5, 5), strides=1, padding='same', activation='relu', name='tonic_cnn_4')(y)
    y = Conv2D(tonic_cnn_filters, (5, 5), strides=1, padding='same', activation='relu', name='tonic_cnn_5')(
        y)  # (1, 60, -1, 128)
    y = Dropout(drop_rate)(y)
    y = tf.squeeze(y, 0)
    y = tf.reduce_mean(y, 1)
    y = Dense(tonic_emb_size, activation='relu', name='tonic_cnn_dense_1')(y)
    return y  # (60,32)


def get_rag_from_rnn(red_y_am, indices, topk, note_emb_add, note_dim, dropout):
    window = tf.minimum(tf.shape(red_y_am)[0], 150)
    slice_ind = tf.random.uniform(shape=(), minval=0, maxval=tf.shape(red_y_am)[0] - window + 1, dtype=tf.int32)

    red_y_am = red_y_am[slice_ind:slice_ind + window]
    red_y_am_all = []
    for i in range(topk):
        red_y_am_all.append(tf.math.mod(red_y_am + indices[i], 60))
    red_y_am_all = tf.stack(red_y_am_all)
    embs = tf.gather(note_emb_add, red_y_am_all)
    if len(embs.shape) == 2:
        embs = tf.expand_dims(embs, 0)
    rnn_1 = Bidirectional(LSTM(note_dim, return_sequences=False, recurrent_dropout=dropout, dropout=dropout))(embs)
    # rnn_1 = Dropout(dropout)(rnn_1)
    # rnn_1 = Bidirectional(LSTM(note_dim, return_sequences=True, recurrent_dropout=dropout, dropout=dropout))(rnn_1)
    # rnn_1 = Dropout(dropout)(rnn_1)
    # rnn_2 = Bidirectional(LSTM(note_dim, recurrent_dropout=dropout, dropout=dropout))(rnn_1)
    # rnn_1 = tf.expand_dims(rnn_1[0],0)

    # out = tf.concat([rnn_1[:,-1,:], rnn_2], axis=1)
    # f = Dense(2*note_dim, activation='sigmoid')(out)
    # out = f*rnn_1[:,-1,:] + (1-f)*rnn_2
    return Dense(2 * note_dim, activation='relu')(rnn_1)


def get_raga_from_transformer(red_y_am, note_emb_add, note_dim, dropout):
    max_len = 1000
    red_y_am = red_y_am[:, :max_len]
    pad_len = max_len - tf.shape(red_y_am)[1]

    # mask_seq = tf.expand_dims(tf.sequence_mask(tf.shape(red_y_am)[1], max_len), 0)
    # mask_seq = tf.tile(mask_seq, [10,1])
    mask_seq = tf.ones_like(red_y_am, tf.float32)

    red_y_am = tf.concat([red_y_am, tf.zeros([10, pad_len], tf.int32)], axis=1)
    mask_seq = tf.concat([mask_seq, tf.zeros([10, pad_len])], axis=1)
    mask_seq = tf.expand_dims(mask_seq, 2)

    raga_enc = Encoder(note_emb_add, enc_num=1, sequence_length=max_len, N=4, size=note_dim)
    encoding = raga_enc.encode(red_y_am, mask_seq, None, True)
    # encoding = encoder_3.encode(red_y_am, note_emb_add, None, mask_seq, 4, d_model=note_dim, dropout=0)
    encoding = tf.reduce_mean(encoding, axis=1)
    # encoding = encoding[:,-1,:]
    return encoding


def min_max_scale(y, is_tonic):
    # mms = (y - tf.reduce_mean(y))/(tf.math.reduce_std(y))
    # tf.clip_by_value()
    # if not is_tonic:
    #     y_min = tf.reduce_min(y)
    #     z = (y - y_min)/(tf.reduce_max(y)-y_min)
    #     z = tf.math.pow(z, 0.5)
    # else:
    #     z = y
    z = y
    # return (y - y_min)/(tf.reduce_max(y)-y_min)
    # mms = tf.clip_by_value(mms, 0, 0.3)
    # y_mean = tf.reduce_mean(y)

    return (z - tf.reduce_mean(z)) / (tf.math.reduce_std(z))


def ffnn(inputs, hidden_size, drop_rate=0.4):
    x = inputs
    for hs in hidden_size:
        den = Dense(hs, activation='relu')(x)
        x = Dropout(drop_rate)(den)
    return x


def get_unique_seq(arg_y):
    # red_y = tf.random.uniform(shape=(100,), maxval=60, dtype=tf.int32)
    # red_y  = tf.one_hot(red_y,60)

    arg_y = tf.concat([[0.], tf.cast(arg_y, tf.float32)], axis=-1)  # None+1

    arg_y_shifted = tf.roll(arg_y, -1, axis=-1)  # 1,None+1

    mask = tf.cast(tf.not_equal(arg_y, arg_y_shifted), tf.float32)  # 1,None+1
    mask = tf.where(mask)[:, 0]
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
    energy = (audio - np.mean(audio)) / np.std(audio)
    energy = np.square(energy)
    energy_frames = as_strided(energy, shape=(1024, n_frames),
                               strides=(energy.itemsize, hop_length * energy.itemsize))
    energy_frames = energy_frames.transpose().copy()
    energy_frames = np.mean(energy_frames, axis=1)
    energy_frames = (energy_frames - np.mean(energy_frames)) / np.std(energy_frames)

    frames = (frames - np.mean(frames, axis=1)) / np.std(frames, axis=1)

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

    return model.predict([frames, mask, chroma, energy_frames], verbose=verbose, batch_size=max_batch_size)


def pad_frames(frames, sequence_length, energy_frames, step_size=10):
    padded_length = sequence_length * np.ceil(len(frames) / sequence_length)
    add_length = int(padded_length) - frames.shape[0]
    add_frames = np.zeros([add_length, 1024]) - 1
    padded_frames = np.concatenate([frames, add_frames], axis=0)

    mask = np.ones(frames.shape[0])
    mask = np.concatenate([mask, np.zeros(add_length)], axis=0)

    energy_frames = np.concatenate([energy_frames, np.zeros(add_length)], axis=0)

    int(max_batch_size * step_size / 1000)
    return padded_frames, energy_frames, mask


def train(task, tradition):
    if task == 'tonic':
        train_tonic(tradition)
    elif task == 'raga':
        # tonic_model_path = train_tonic(tradition)
        raga_model_path = 'model/{}_raga_model.hdf5'.format(tradition)

        config = pyhocon.ConfigFactory.parse_file("experiments.conf")[task]
        training_generator = DataGenerator(task, tradition, 'train', config, random=False)
        validation_generator = DataGenerator(task, tradition, 'validate', config, random=False)
        model = build_and_load_model(config, task)
        # model.load_weights(tonic_model_path, by_name=True)
        # model.load_weights('model/model-full.h5', by_name=True)
        # model.fit(generator)
        # model.fit(x=training_generator,
        #                     validation_data=validation_generator, verbose=2, epochs=15, shuffle=True, batch_size=1)

        checkpoint = ModelCheckpoint(raga_model_path, monitor='loss', verbose=1,
                                     save_best_only=True, mode='auto', period=1)
        # early_stopping_callback = EarlyStopping(monitor="val_loss", patience=5, mode="auto", restore_best_weights=True, min_delta=0.0000001)
        model.fit_generator(generator=training_generator,
                            validation_data=validation_generator, verbose=1, epochs=50, shuffle=True,
                            callbacks=[checkpoint])


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
                        validation_data=validation_generator, verbose=1, epochs=15, shuffle=True,
                        callbacks=[checkpoint])

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
    # print(np.max(p, axis=1))
    # print(p[0])
    print(np.argmax(p[0], axis=1))
    # print(np.max(p[1], axis=1))
    # print(np.argmax(p[1], axis=1))
    # for pi in p[0]:
    #     print(pi)
    cents = to_local_average_cents(p[1])
    frequency = 10 * 2 ** (cents / 1200)

    for f in frequency:
        print(f)
    # print('pred', frequency)


def test_pitch(task, tradition):
    config = pyhocon.ConfigFactory.parse_file("experiments.conf")[task]
    model = build_and_load_model(config, task)
    model.load_weights('model/model-full.h5', by_name=True)
    model_srate = config['model_srate']
    step_size = config['hop_size']
    cuttoff = config['cutoff']
    for t in ['train', 'validate', 'test']:
        data_path = config[tradition + '_' + t]
        data = pd.read_csv(data_path, sep='\t')
        data = data.reset_index()

        slice_ind = 0
        k = 0

        while k < data.shape[0]:
            path = data.loc[k, 'path']
            # path = 'E:\\E2ERaga\\data\\RagaDataset\\audio\\f5999e30-d00d-4837-b9c6-5328768ae22d.wav'
            pitch_path = path[:path.index('.wav')] + '.pitch'
            pitch_path = pitch_path.replace('audio', 'pitches')

            pitch_file = open(pitch_path, "w")
            # if os.path.exists(pitch_path):
            #     pitch_file = open(pitch_path, "a")
            # else:
            #     pitch_file = open(pitch_path, "w")
            pitches = []
            while True:
                if slice_ind == 0:
                    print(pitch_path)

                frames, slice_ind = __data_generation_pitch(path, slice_ind, model_srate, step_size, cuttoff)

                p = model.predict(np.array([frames]))
                # p = np.sum(np.reshape(p, [-1,6,60]), axis=1)
                cents = to_local_average_cents(p)
                frequency = 10 * 2 ** (cents / 1200)
                # for p1 in p:
                #     p1 = list(map(str, p1))
                #     p1 = ','.join(p1)
                #     pitches.append(p1)
                # p = ','.join(p)
                pitches.extend(frequency)
                # pitches.extend(p)
                # pitches.append(p)
                if slice_ind == 0:
                    k += 1
                    break
                # frequency = list(map(str, frequency))
            pitches = list(map(str, pitches))
            pitch_file.writelines('\n'.join(pitches))
            pitch_file.close()
            break
        break


def cache_cqt(task, tradition):
    config = pyhocon.ConfigFactory.parse_file("experiments.conf")[task]

    with h5py.File('data/RagaDataset/Hindustani/cqt_cache.hdf5', "w") as f:
        for t in ['train', 'validate', 'test']:
            data_path = config[tradition + '_' + t]
            data = pd.read_csv(data_path, sep='\t')
            data = data.reset_index()
            k = 0
            while k < data.shape[0]:
                path = data.loc[k, 'path']
                mbid = data.loc[k, 'mbid']
                cqt = get_cqt(path)
                f.create_dataset(mbid, data=cqt)
                print(mbid, k)
                k += 1


def get_cqt(path):
    # return tf.ones([1,60], np.float32)

    sr, audio = wavfile.read(path)
    if len(audio.shape) == 2:
        audio = audio.mean(1)  # make mono
    C = np.abs(librosa.cqt(audio, sr=sr, bins_per_octave=60, n_bins=60 * 7, pad_mode='wrap',
                           fmin=librosa.note_to_hz('C1')))
    #     librosa.display.specshow(C, sr=sr,x_axis='time', y_axis='cqt', cmap='coolwarm')

    # fig, ax = plt.subplots()
    c_cqt = librosa.amplitude_to_db(C, ref=np.max)
    c_cqt = np.reshape(c_cqt, [7, 60, -1])
    c_cqt = np.mean(c_cqt, axis=0)
    # c_cqt = np.mean(c_cqt, axis=1, keepdims=True)
    # img = librosa.display.specshow(c_cqt,
    #                                sr=self.model_srate, x_axis='time', y_axis='cqt_note', ax=ax, bins_per_octave=60)
    # ax.set_title('Constant-Q power spectrum')
    # fig.colorbar(img, ax=ax, format="%+2.0f dB")
    return c_cqt


def __data_generation_pitch(path, slice_ind, model_srate, step_size, cuttoff):
    # pitch_path = self.data.loc[index, 'pitch_path']
    # if self.current_data[2] == path:
    #     frames = self.current_data[0]
    #     pitches = self.current_data[1]
    #     pitches = pitches[slice_ind * int(len(frames) / n_cutoff):(slice_ind + 1) * int(len(frames) / n_cutoff)]
    #     frames = frames[slice_ind * int(len(frames) / n_cutoff):(slice_ind + 1) * int(len(frames) / n_cutoff)]
    #     return frames, pitches
    # else:
    #     sr, audio = wavfile.read(path)
    #     if len(audio.shape) == 2:
    #         audio = audio.mean(1)  # make mono
    #     audio = self.get_non_zero(audio)
    sr, audio = wavfile.read(path)
    if len(audio.shape) == 2:
        audio = audio.mean(1)  # make mono
    # audio = self.get_non_zero(audio)

    # audio = audio[:self.model_srate*15]
    # audio = self.mp3_to_wav(path)

    # print(audio[:100])
    audio = np.pad(audio, 512, mode='constant', constant_values=0)
    audio_len = len(audio)
    audio = audio[slice_ind * model_srate * cuttoff:(slice_ind + 1) * model_srate * cuttoff]
    if (slice_ind + 1) * model_srate * cuttoff >= audio_len:
        slice_ind = -1
    # audio = audio[: self.model_srate*self.cutoff]
    hop_length = int(model_srate * step_size)
    n_frames = 1 + int((len(audio) - 1024) / hop_length)
    frames = as_strided(audio, shape=(1024, n_frames),
                        strides=(audio.itemsize, hop_length * audio.itemsize))
    frames = frames.transpose().copy()
    frames -= np.mean(frames, axis=1)[:, np.newaxis]
    frames /= (np.std(frames, axis=1)[:, np.newaxis] + 1e-5)

    return frames, slice_ind + 1


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
    zarg = zarg % 60
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

    time, frequency, cents, confidence, activation = predict(
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

