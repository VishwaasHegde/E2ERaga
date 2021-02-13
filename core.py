from __future__ import division
from __future__ import print_function

# import re
import sys
import math
from scipy.io import wavfile
import numpy as np
from numpy.lib.stride_tricks import as_strided
import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Conv2D, BatchNormalization, Softmax, Conv1D, Bidirectional
from tensorflow.keras.layers import MaxPool2D, Dropout, Permute, Flatten, Dense, MaxPool1D, AvgPool1D, Bidirectional, \
    LSTM, Lambda
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
import librosa
import librosa.display
import pyhocon
import os
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import recorder
from scipy import stats
from pydub import AudioSegment
from resampy import resample

# store as a global variable, since we only support a few models for now
from data_generator import DataGenerator

models = {
    'tiny': None,
    'small': None,
    'medium': None,
    'large': None,
    'full': None
}

# for layer in res_net.layers[:143]:
#     layer.trainable = False
# for layer in res_net.layers[143:]:
#     layer.trainable = True


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
    pitches_batch = Input(shape=(None, 60), name='pitches_input', dtype='float32')
    random_batch = Input(shape=(), name='random_input', dtype='int32')
    cqt_batch = Input(shape=(None,60), name='cqt_input', dtype='float32')
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

    # red_y = tf.reshape(pitches, [-1, 6, 60])
    # red_y = tf.reduce_sum(red_y, axis=1)  # (None, 60)
    red_y = pitches
    red_y = tf.roll(red_y, -transpose_by, axis=1)
    cqt = tf.roll(cqt, -transpose_by, axis=-1)
    # red_y_random = apply_note_random(red_y)
    # red_y = random*red_y_random + (1-random)*red_y
    # Tonic
    histograms = get_histograms(red_y, cqt)
    tonic_logits = get_tonic_emb(red_y, cqt, histograms, note_emb, note_dim, drop_rate_tonic)

    raga_emb = get_raga_emb(red_y, cqt, histograms, tonic_logits, note_emb, note_dim, random, drop_rate_raga)
    # raga_logits = get_raga_emb(red_y, cqt, tonic_logits, note_emb, note_dim, drop_rate_raga)
    raga_logits = Dense(n_labels, activation='softmax', name='raga')(raga_emb)

    loss_weights = config['loss_weights']

    tonic_logits_roll = tf.roll(tonic_logits, transpose_by, axis=-1, name='tonic')

    rag_model = Model(inputs=[pitches_batch, random_batch, cqt_batch], outputs=[raga_logits, tonic_logits_roll])
    cce = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.0)
    # rag_model.compile(loss={'raga': cce, 'tf_op_layer_tonic': 'binary_crossentropy'},
    #                   optimizer='adam', metrics={'raga': 'accuracy', 'tf_op_layer_tonic': 'accuracy'},
    #                   loss_weights={'raga': loss_weights[0], 'tf_op_layer_tonic': loss_weights[1]})

    rag_model.compile(loss={'raga': cce, 'tf_op_layer_tonic': 'binary_crossentropy'},
                      optimizer='adam', metrics={'raga': 'accuracy', 'tf_op_layer_tonic': 'accuracy'},
                      loss_weights={'raga': loss_weights[0], 'tf_op_layer_tonic': loss_weights[1]})

    # rag_model.load_weights('model/carnatic_raga_model.hdf5', by_name=True, skip_mismatch=True)
    rag_model.load_weights('model/hindustani_raga_model.hdf5', by_name=True, skip_mismatch=True)

    # rag_model.compile(loss={'raga': 'categorical_crossentropy'},
    #                   optimizer='adam', metrics={'raga': 'accuracy'},
    #                   loss_weights={'raga': loss_weights[0]})

    # rag_model.summary()
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


def get_hist_emb(histograms, note_dim, indices, topk, is_tonic, drop_rate=0.2):
    hist_cc = [tf.cast(standardize(h), tf.float32) for h in histograms]

    # for idx, h in enumerate(hist_cc):
    #     if idx >= 2:
    #         h = AvgPool1D(pool_size=3, strides=1, padding='same')(tf.expand_dims(tf.expand_dims(h, 0), 2))[0, :, 0]
    #     hist_cc[idx] = h

    hist_cc = tf.transpose(tf.stack(hist_cc))
    hist_cc_all = []
    for i in range(topk):
        hist_cc_trans = tf.roll(hist_cc, -indices[i], axis=0)
        hist_cc_all.append(hist_cc_trans)
    hist_cc_all = tf.stack(hist_cc_all)

    if is_tonic:
        hist_cc_all = Dropout(0.3)(hist_cc_all)
    else:
        hist_cc_all = Dropout(0.1)(hist_cc_all)

    d = 1

    if is_tonic:
        f=64
    else:
        f=64

    z = convolution_block(d, hist_cc_all, 5, f, drop_rate)
    z = convolution_block(d, z, 3, 2*f, drop_rate)
    z = convolution_block(d, z, 3, 4*f, drop_rate)

    z = Flatten()(z)
    z = Dense(2 * note_dim, activation='relu')(z)
    return z

def get_tonic_emb(red_y, cqt, histograms, note_emb, note_dim, drop_rate=0.2):
    topk=9

    indices = tf.random.uniform(shape=(topk,), minval=0, maxval=60, dtype=tf.int32)
    hist_emb = get_hist_emb(histograms, note_dim, indices, topk, True, drop_rate)

    cqt_emb = get_tonic_from_cqt(cqt, indices, topk, tonic_emb_size=note_dim, tonic_cnn_filters=128, drop_rate=drop_rate)

    tonic_emb = combine(hist_emb, cqt_emb, topk, note_dim, drop_rate, True)

    tonic_emb = hist_emb

    tonic_logits_norm = []
    tonic_logits = Dense(60, activation='sigmoid')(tonic_emb)
    for i in range(topk):
        tonic_logits_norm.append(tf.roll(tonic_logits[i],indices[i], axis=0))
    tonic_logits = tf.reduce_mean(tonic_logits_norm, axis=0, keepdims=True)

    return tonic_logits


def get_raga_emb(red_y, cqt, histograms, tonic_logits, note_emb, note_dim, random, drop_rate=0.2):
    topk = 9
    rs = topk//2
    re = (topk//2) + 1


    tonic_argmax = tf.argmax(tonic_logits[0])
    indices = tf.range(tonic_argmax - rs, tonic_argmax + re)
    indices = tf.math.mod(60 + indices, 60)
    values = tf.gather(tonic_logits[0], indices)

    hist_emb = get_hist_emb(histograms, note_dim, indices, topk, False, drop_rate)

    ndms_input_1 = get_ndms_input(red_y, cqt, histograms, indices, topk, False, random, False)
    ndms_input_2 = get_ndms_input(tf.reverse(red_y, axis=[0]), tf.reverse(cqt, axis=[0]), histograms, indices, topk, False, random, True)

    # ndms_input = ndms_input_1
    # ndms_input = tf.concat([ndms_input_1, ndms_input_2], axis=0)
    ndms_input = (ndms_input_1 + ndms_input_2)/2
    # ndms_input = ndms_input_1

    ndms = get_ndms(ndms_input, note_dim, drop_rate=drop_rate)

    # ndms = (ndms[:topk] + ndms[topk:])/2

    raga_emb = combine(hist_emb, ndms, topk, note_dim, drop_rate, False)
    # raga_emb = hist_emb
    raga_emb = tf.reduce_sum(tf.multiply(raga_emb, tf.expand_dims(values, 1)), axis=0, keepdims=True) / tf.reduce_sum(
        values)

    return raga_emb


def apply_note_random(red_y):
    def apply_random_roll(a):
        r = tf.random.uniform(shape=(), minval=-2, maxval=3, dtype=tf.int32)
        return tf.roll(a, r, axis=-1)

    red_y_roll = tf.keras.layers.Lambda(apply_random_roll)(red_y)
    return red_y_roll


def combine(emb1, emb2, topk, note_dim, drop_rate, is_tonic, id=1):
    if is_tonic:
        emb = tf.transpose(tf.concat([emb1,emb2], axis=0))
        emb = Dropout(0.2)(emb)
        emb = Dense(topk)(emb)
        return tf.transpose(emb)
    else:

        f = Dense(1, activation='sigmoid')(tf.concat([emb1], axis=1))
        # f=0
        emb = f * emb1+ (1-f) * emb2

        return emb

def convolution_block(d, input_data, ks, f, drop_rate):
    if d==1:
        z = Conv1D(filters=f, kernel_size=ks, strides=1, padding='same', activation='relu')(input_data)
        z = Conv1D(filters=f, kernel_size=ks, strides=1, padding='same', activation='relu')(z)
        z = BatchNormalization()(z)
        z = MaxPool1D(pool_size=2)(z)
        z = Dropout(drop_rate)(z)
    else:
        z = Conv2D(filters=f, kernel_size=(ks, ks), strides=(1, 1), activation='relu', padding='same')(input_data)
        z = Conv2D(filters=f, kernel_size=(ks, ks), strides=(1, 1), activation='relu', padding='same')(z)
        z = BatchNormalization()(z)
        z = MaxPool2D(pool_size=(2, 2), strides=None, padding='valid')(z)
        z = Dropout(drop_rate)(z)

    return z


def get_histograms(red_y, cqt):
    red_y_mean = tf.math.reduce_mean(red_y, 0)
    red_y_std = tf.math.reduce_std(red_y, 0)
    cqt_mean = tf.math.reduce_mean(cqt, 0)
    cqt_std = tf.math.reduce_std(cqt, 0)

    return [red_y_mean, red_y_std, cqt_mean, cqt_std]




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


def get_ndms_input(red_y, cqt, histograms, indices, topk, is_tonic, random=False, flip=False):
    c_note = freq_to_cents(31.7 * 2, 40)
    c_note = np.reshape(c_note, [6, 60])
    c_note = np.sum(c_note, axis=0)
    c_note_5 = c_note
    #     c_note_5 = tf.reduce_mean([tf.roll(c_note, i, 0) for i in range(-2, 3)], axis=0)
    c_note_5 = normalize(c_note_5)
    # c_note_5 = tf.constant([1,0,0,0,0,0,0,0,0,0,0,0], dtype=tf.float64)
    diag_notes = tf.keras.layers.Lambda(
        lambda x: tf.map_fn(lambda a: tf.roll(tf.constant(c_note_5), a, axis=-1), x, tf.float64))(tf.range(60))
    diag_notes = tf.cast(diag_notes, tf.float32)
    diag_notes = 1 - normalize(diag_notes)

    histograms = [normalize(h) for h in histograms]
    histograms_mat = [tf.matmul(tf.pow(tf.expand_dims(h, 0), 0.5), tf.pow(tf.expand_dims(h, 0), 0.5), transpose_a=True)
                      for h in histograms]
    histograms_mat = [normalize(tf.multiply(normalize(tf.cast(h, tf.float32)), 1 - diag_notes)) for h in histograms_mat]
    # histograms_mat = [tf.roll(h, [3, 3], axis=[0, 1]) for h in histograms_mat]

    c_note = freq_to_cents(31.7 * 2, 100)
    c_note = np.reshape(c_note, [6, 60])
    c_note = np.sum(c_note, axis=0)
    c_note_5 = c_note
    #     c_note_5 = tf.reduce_mean([tf.roll(c_note, i, 0) for i in range(-2, 3)], axis=0)
    c_note_5 = normalize(c_note_5)
    # c_note_5 = tf.constant([1,0,0,0,0,0,0,0,0,0,0,0], dtype=tf.float64)
    diag_notes = tf.keras.layers.Lambda(
        lambda x: tf.map_fn(lambda a: tf.roll(tf.constant(c_note_5), a, axis=-1), x, tf.float64))(tf.range(60))
    diag_notes = tf.cast(diag_notes, tf.float32)
    diag_notes = 1 - normalize(diag_notes)


    top_notes = tf.range(2, 60, 5)
    top_notes = tf.reduce_sum(tf.one_hot(top_notes, 60), 0)

    matmuls = []
    red_y_clustered = cluster_top_notes(red_y, top_notes)
    red_y_am, red_y_am_val = get_unique_seq_1(red_y_clustered, red_y)
    matmul = 0
    for s in range(1, 5):
        matmul += get_ndms_mat(red_y_am, red_y_am_val, -1 * s)

    for idx, hist in enumerate(histograms[:4]):
        hist = normalize(hist)
        hist_exp = tf.expand_dims(hist, 0)
        hist_exp = tf.math.pow(hist_exp, 0.5)
        hist_stand = normalize(tf.matmul(hist_exp, hist_exp, transpose_a=True))

        diag_notes_stand = normalize(diag_notes)
        temp_mat = matmul
        temp_mat = normalize(normalize(tf.cast(temp_mat, tf.float32)) * normalize(hist_stand * diag_notes_stand))

        temp_mat = normalize(normalize(tf.cast(temp_mat, tf.float32) * diag_notes_stand) + normalize(histograms_mat[int(tf.math.mod(idx,4))]))

        temp_mat = normalize(temp_mat)
        temp_mat = standardize(temp_mat)
        if flip:
            temp_mat = tf.transpose(temp_mat)

        matmuls.append(temp_mat)
        # matmuls.append(temp_mat)

    # matmuls.append(matmuls[0])
    # matmuls.append(matmuls[1])

    matmul = tf.stack(matmuls, axis=2)
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

        ndms = Dropout(0.2)(ndms)

    return ndms


def get_ndms(ndms, note_dim, drop_rate=0.2):
    d = 2
    f = 64

    z = convolution_block(d, ndms, 5, f, drop_rate)
    z = convolution_block(d, z, 3, 2*f, drop_rate)
    z = convolution_block(d, z, 3, 4*f, drop_rate)
    z = Flatten()(z)
    z = Dropout(drop_rate)(z)
    z = Dense(2 * note_dim, activation='relu')(z)
    return z

def cluster_top_notes(red_y, top_notes):
    # c_note = freq_to_cents(31.7 * 2)
    # c_note = np.reshape(c_note, [6, 60])
    # c_note = np.sum(c_note, axis=0)

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
    return top_notes_where


def get_unique_seq_1(arg_y, red_y):
    # red_y = tf.random.uniform(shape=(100,), maxval=60, dtype=tf.int32)
    # red_y  = tf.one_hot(red_y,60)
    red_y_p = tf.reduce_max(red_y, axis=1)

    arg_y = tf.concat([arg_y, [60]], axis=-1)  # None+1
    arg_y_shifted = tf.roll(arg_y, 1, axis=-1)  # 1,None
    arg_y = arg_y[:-1]
    arg_y_shifted = arg_y_shifted[:-1]

    mask = tf.cast(tf.not_equal(arg_y, arg_y_shifted), tf.float32)  # 1,None+1

    mask_cumsum = tf.math.cumsum(mask)-1
    mask_cumsum = tf.math.segment_sum(red_y_p, tf.cast(mask_cumsum, tf.int32))
#     mask_cumsum = tf.math.segment_sum(np.ones_like(red_y_p), tf.cast(mask_cumsum, tf.int32))
    mask = tf.where(mask)[:, 0]
    # mask_pad = tf.concat([[0], tf.cast(mask, tf.int32)], axis=0)
    # mask_roll = tf.roll(mask_pad, -1, axis=-1)
    # mask_roll = tf.math.abs(mask_roll - mask_pad)[:-1]
    #     print(mask_roll)
    uni_seq_notes = tf.gather(arg_y, mask)
    uni_seq_notes = tf.cast(uni_seq_notes, tf.int32)
    return uni_seq_notes, mask_cumsum


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
    rs = topk // 2
    re = topk // 2 + 1
    values = pred_tonic[60 + bces_argmin - rs:60 + bces_argmin + re]

    indices = tf.range(bces_argmin - rs, bces_argmin + re)
    indices = tf.math.mod(60 + indices, 60)
    return values, indices


def get_ndms_mat(ndms, ndms_val, shift=-1):
    c_note = freq_to_cents(65.41,25)
    c_note = np.reshape(c_note, [6,60])
    c_note = np.sum(c_note,axis=0)
    ndms_ohe = tf.keras.layers.Lambda(lambda x: tf.map_fn(lambda a: tf.roll(tf.constant(c_note), a, axis=-1), x, tf.float64))(ndms)
    ndms_roll_ohe = tf.roll(ndms_ohe, shift, axis=0)
#     print(ndms_val)
    ndms_val = tf.cast(tf.expand_dims(ndms_val,1), tf.float64)
    ndms_ohe = tf.multiply(ndms_ohe, ndms_val)
    ndms_roll_ohe = tf.multiply(ndms_roll_ohe, tf.roll(ndms_val, shift, axis=0))
    # ndms_roll_ohe = tf.keras.layers.Lambda(lambda x: tf.map_fn(lambda a: tf.roll(tf.constant(c_note), a, axis=-1), x, tf.float64))(ndms_roll)
    # ndms_roll_ohe = tf.roll(ndms_ohe, -1, axis=0)
    # ndms_ohe = tf.one_hot(ndms, 60)
    # ndms_roll = tf.roll(ndms, -1, axis=-1)
    # ndms_roll_ohe = tf.one_hot(ndms_roll, 60)
    matmul = tf.matmul(ndms_ohe, ndms_roll_ohe, transpose_a=True)

    return matmul


def get_tonic_from_cqt(cqt, indices, topk, tonic_emb_size=128, tonic_cnn_filters=128, drop_rate=0.1):
    # chroma, int(note_dim*3/32), int(note_dim/2), drop_rate
    cut_off = 100

    cqts = []
    for i in range(topk):
        slice_ind = tf.random.uniform(shape=(), minval=0, maxval=tf.shape(cqt)[0]-cut_off, dtype=tf.int32)
        cqt_slice = cqt[slice_ind:slice_ind+cut_off]
        cqt_slice = tf.roll(cqt_slice, -indices[i], axis=0)
        cqts.append(cqt_slice)

    cqts = tf.stack(cqts)
    cqts = tf.expand_dims(cqts, 3)

    y = Conv2D(tonic_cnn_filters, (5, 5), strides=1, padding='same', activation='relu', name='tonic_cnn_1')(cqts)
    y = Conv2D(tonic_cnn_filters, (5, 5), strides=1, padding='same', activation='relu', name='tonic_cnn_2')(y)
    y = Conv2D(tonic_cnn_filters, (5, 5), strides=1, padding='same', activation='relu', name='tonic_cnn_3')(y)
    y = Dropout(drop_rate)(y)
    y = Conv2D(tonic_cnn_filters, (5, 5), strides=1, padding='same', activation='relu', name='tonic_cnn_4')(y)
    y = Conv2D(tonic_cnn_filters, (5, 5), strides=1, padding='same', activation='relu', name='tonic_cnn_5')(
        y)  # (1, 60, -1, 128)
    y = Dropout(drop_rate)(y)
    # y = tf.squeeze(y, 1)
    y = tf.reduce_mean(y, 1)
    y = Flatten()(y)
    y = Dense(2*tonic_emb_size, activation='relu')(y)

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
    return Dense(2 * note_dim, activation='relu')(rnn_1)


def get_raga_from_transformer(red_y_am, note_emb_add, note_dim, dropout):
    max_len = 1000
    red_y_am = red_y_am[:, :max_len]
    pad_len = max_len - tf.shape(red_y_am)[1]
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


def standardize(z):
    return (z - tf.reduce_mean(z)) / (tf.math.reduce_std(z))

def normalize(z):
    min_z = tf.reduce_min(z)
    return (z - min_z) / (tf.reduce_max(z) - min_z)


def ffnn(inputs, hidden_size, drop_rate):
    x = inputs
    for idx, hs in enumerate(hidden_size):
        den = Dense(hs, activation='relu')(x)
        x = Dropout(drop_rate[idx])(den)
    return x


def get_unique_seq(arg_y):
    arg_y = tf.concat([[0.], tf.cast(arg_y, tf.float32)], axis=-1)  # None+1

    arg_y_shifted = tf.roll(arg_y, -1, axis=-1)  # 1,None+1

    mask = tf.cast(tf.not_equal(arg_y, arg_y_shifted), tf.float32)  # 1,None+1
    mask = tf.where(mask)[:, 0]
    uni_seq_notes = tf.gather(arg_y_shifted, mask)
    uni_seq_notes = tf.cast(uni_seq_notes, tf.int32)
    return uni_seq_notes


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


def train(task, tradition):
    if task == 'tonic':
        train_tonic(tradition)
    elif task == 'raga':
        # tonic_model_path = train_tonic(tradition)
        raga_model_path = 'model/{}_raga_model.hdf5'.format(tradition)

        config = pyhocon.ConfigFactory.parse_file("experiments.conf")[task]
        training_generator = DataGenerator(task, tradition, 'train', config, random=True, shuffle=True, full_length=True)
        validation_generator = DataGenerator(task, tradition, 'validate', config, random=False, shuffle=False, full_length=False)
        test_generator = DataGenerator(task, tradition, 'test', config, random=False, shuffle=False, full_length=False)
        model = build_and_load_model(config, task)
        # model.load_weights(tonic_model_path, by_name=True)
        # model.load_weights('model/model-full.h5', by_name=True)
        # model.fit(generator)
        # model.fit(x=training_generator,
        #                     validation_data=validation_generator, verbose=2, epochs=15, shuffle=True, batch_size=1)

        checkpoint = ModelCheckpoint(raga_model_path, monitor='val_raga_accuracy', verbose=1,
                                     save_best_only=True, mode='max', period=1)
        # early_stopping_callback = EarlyStopping(monitor="val_loss", patience=5, mode="auto", restore_best_weights=True, min_delta=0.0000001)
        # model.fit_generator(generator=training_generator,
        #                     validation_data=validation_generator, verbose=1, epochs=50, shuffle=True,
        #                     callbacks=[checkpoint])
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
    test_generator = DataGenerator(task, tradition, 'test', config, random=True, shuffle=False, full_length=True)
    # test_generator = DataGenerator(task, tradition, 'validate', config, random=True, shuffle=False, full_length=True)
    # test_generator = DataGenerator(task, tradition, 'validate', config, random=True, shuffle=False,
    #                                      full_length=True)
    # test_generator = DataGenerator(task, tradition, 'test', config, random=True)
    model = build_and_load_model(config, task)
    # model.fit(generator)
    # model.fit(x=training_generator,
    #                     validation_data=validation_generator, verbose=2, epochs=15, shuffle=True, batch_size=1)
    # model.load_weights('model/hindustani_tonic_model.hdf5', by_name=True)
    # model.load_weights('model/model-full.h5'.format(tradition, 'tonic'), by_name=True)
    # model.load_weights('model/{}_{}_model.hdf5'.format(tradition, 'tonic'), by_name=True)
    model.load_weights('model/{}_raga_model.hdf5'.format(tradition), by_name=True)
    data = test_generator.data
    raga_pred = None
    tonic_pred = None
    all_p = []
    for i in range(5):
        p = model.predict_generator(test_generator, verbose=1)
        all_p.append(np.argmax(p[0], axis=1))

        # print(np.argmax(p[0], axis=1))
        if raga_pred is None:
            raga_pred = p[0]
            tonic_pred = p[1]
        else:
            raga_pred += p[0]
            tonic_pred += p[1]
    # all_p = np.array(all_p)
    print(stats.mode(all_p, axis=0))
    data['predicted_raga'] = np.argmax(raga_pred, axis=1)

    acc = 0
    print(data['predicted_raga'].values)
    for t,p in zip(data['labels'].values, data['predicted_raga'].values):
        acc += int(t == p)
    print('raga accuracy: {}'.format(acc/data.shape[0]))
    # model.evaluate_generator(generator=test_generator, verbose=1)
    # p = model.predict_generator(test_generator, verbose=1)
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
    # print(np.argmax(p[0], axis=1))
    # print(np.max(p[1], axis=1))
    # print(np.argmax(p[1], axis=1))
    # for pi in p[0]:
    #     print(pi)
    cents = to_local_average_cents(tonic_pred)
    frequency = 10 * 2 ** (cents / 1200)
    #
    data['predicted_tonic'] = frequency
    data.to_csv(config[tradition+'_'+'output'], index=False, sep='\t')
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
            # path = 'E:\\E2ERaga\\data\\RagaDataset\\audio\\cb280397-4e13-448e-9bd7-97105b2347dc.wav'
            pitch_path = path[:path.index('.wav')] + '.pitch'
            pitch_path = pitch_path.replace('audio', 'pitches')

            if os.path.exists(pitch_path):
                k+=1
                continue

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
                p = np.sum(np.reshape(p, [-1,6,60]), axis=1)
                # cents = to_local_average_cents(p)
                # frequency = 10 * 2 ** (cents / 1200)
                for p1 in p:
                    p1 = list(map(str, p1))
                    p1 = ','.join(p1)
                    pitches.append(p1)
                # p = ','.join(p)
                # pitches.extend(frequency)
                # pitches.extend(p)
                # pitches.append(p)
                if slice_ind == 0:
                    k += 1
                    break
                # frequency = list(map(str, frequency))
            pitches = list(map(str, pitches))
            pitch_file = open(pitch_path, "w")
            pitch_file.writelines('\n'.join(pitches))
            pitch_file.close()

def predict_run_time(tradition, seconds=60):
    pitch_config = pyhocon.ConfigFactory.parse_file("experiments.conf")['pitch']
    pitch_model = build_and_load_model(pitch_config, 'pitch')
    pitch_model.load_weights('model/model-full.h5', by_name=True)

    raga_config = pyhocon.ConfigFactory.parse_file("experiments.conf")['raga']
    raga_model = build_and_load_model(raga_config, 'raga')
    raga_model.load_weights('model/{}_raga_model.hdf5'.format(tradition), by_name=True)

    while True:
        audio = recorder.record(seconds)
        if audio is None:
            return
        get_raga_tonic_prediction(audio, pitch_config, pitch_model, raga_model)


def get_raga_tonic_prediction(audio, pitch_config, pitch_model, raga_model):
    frames = audio_2_frames(audio, pitch_config)
    p = pitch_model.predict(np.array([frames]))
    cents = to_local_average_cents(p)
    frequencies = 10 * 2 ** (cents / 1200)
    pitches = [freq_to_cents(freq) for freq in frequencies]
    pitches = np.expand_dims(pitches, 0)
    cqt = get_cqt(audio)
    p = raga_model.predict([pitches, np.array([False]), cqt])
    cents = to_local_average_cents(p[1])
    frequency = 10 * 2 ** (cents / 1200)
    print('raga: {}'.format(np.argmax(p[0])))
    print('tonic: {}'.format(frequency[0]))


def predict_run_time_file(file_path, tradition, filetype='wav'):

    pitch_config = pyhocon.ConfigFactory.parse_file("experiments.conf")['pitch']
    pitch_model = build_and_load_model(pitch_config, 'pitch')
    pitch_model.load_weights('model/model-full.h5', by_name=True)

    raga_config = pyhocon.ConfigFactory.parse_file("experiments.conf")['raga']
    raga_model = build_and_load_model(raga_config, 'raga')
    raga_model.load_weights('model/{}_raga_model.hdf5'.format(tradition), by_name=True)

    if filetype == 'mp3':
        audio = mp3_to_wav(file_path)
    else:
        sr, audio = wavfile.read(file_path)
        if len(audio.shape) == 2:
            audio = audio.mean(1)

    get_raga_tonic_prediction(audio, pitch_config, pitch_model, raga_model)


def mp3_to_wav(mp3_path):

    a = AudioSegment.from_mp3(mp3_path)
    y = np.array(a.get_array_of_samples())

    if a.channels == 2:
        y = y.reshape((-1, 2))
        y = y.mean(1)
    y = np.float32(y) / 2 ** 15

    y = resample(y, a.frame_rate, 16000)
    return y


def audio_2_frames(audio, config):
    # audio = audio[: self.model_srate*self.cutoff]
    hop_length = int(16000 * config['hop_size'])
    n_frames = 1 + int((len(audio) - 1024) / hop_length)
    frames = as_strided(audio, shape=(1024, n_frames),
                        strides=(audio.itemsize, hop_length * audio.itemsize))
    frames = frames.transpose().copy()
    frames -= np.mean(frames, axis=1)[:, np.newaxis]
    frames /= (np.std(frames, axis=1)[:, np.newaxis] + 1e-5)

    return frames



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


def get_cqt(audio, sr=16000):
    C = np.abs(librosa.cqt(audio, sr=sr, bins_per_octave=60, n_bins=60 * 7, pad_mode='wrap',
                           fmin=librosa.note_to_hz('C1')))
    #     librosa.display.specshow(C, sr=sr,x_axis='time', y_axis='cqt', cmap='coolwarm')

    # fig, ax = plt.subplots()
    c_cqt = librosa.amplitude_to_db(C, ref=np.max)
    c_cqt = np.reshape(c_cqt, [7, 60, -1])
    c_cqt = np.mean(c_cqt, axis=0)
    c_cqt = np.transpose(c_cqt)
    c_cqt = np.expand_dims(c_cqt,0)
    # c_cqt = np.mean(c_cqt, axis=1, keepdims=True)
    # img = librosa.display.specshow(c_cqt,
    #                                sr=self.model_srate, x_axis='time', y_axis='cqt_note', ax=ax, bins_per_octave=60)
    # ax.set_title('Constant-Q power spectrum')
    # fig.colorbar(img, ax=ax, format="%+2.0f dB")
    return c_cqt


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

