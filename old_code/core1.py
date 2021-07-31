from __future__ import division
from __future__ import print_function

import os
import re
import sys
import math
from crepe.data_generator import DataGenerator
from scipy.io import wavfile
import numpy as np
from numpy.lib.stride_tricks import as_strided
import tensorflow as tf
# tf.config.run_functions_eagerly(True)
# tf.compat.v1.disable_eager_execution()
from tensorflow.keras.layers import Input, Reshape, Conv2D, BatchNormalization, Softmax, Conv1D, Bidirectional
from tensorflow.keras.layers import MaxPool2D, Dropout, Permute, Flatten, Dense, MaxPool1D, AvgPool1D
from tensorflow.keras.models import Model
import librosa
import librosa.display
from crepe import encoder_2
import pyhocon
import os
import json
import pandas as pd
from collections import defaultdict
from tensorflow.keras.callbacks import ModelCheckpoint


import matplotlib.pyplot as plt
# store as a global variable, since we only support a few models for now
from crepe.data_generator import DataGenerator

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

def build_and_load_model(config, to_train = 'raga'):
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
    N = config['encoder_layers']
    drop_rate = config['drop_rate']
    tonic_filter = config['tonic_filter']
    tonic_emb_size = config['tonic_emb_size']
    tonic_cnn_filters = config['tonic_cnn_filters']
    n_labels = config['n_labels']
    
    note_dim = config['note_dim']

    if models[model_capacity] is None:
        capacity_multiplier = {
            'tiny': 4, 'small': 8, 'medium': 16, 'large': 24, 'full': 32
        }[model_capacity]

        layers = [1, 2, 3, 4, 5, 6]
        filters = [n * capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
        widths = [512, 64, 64, 64, 64, 64]
        strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

        # x = Input(shape=(1024,), name='input2', dtype='float32')
        x_batch = Input(shape=(None, 1024), name='x_input', dtype='float32')
        mask_batch = Input(shape=(None,), name='mask_input', dtype='int32')
        chroma_batch = Input(shape=(60, None), name='chroma_input', dtype='float32')
        energy_batch = Input(shape=(None,), name='energy_input', dtype='float32')


        x = x_batch[0]
        chroma = chroma_batch[0]
        mask = mask_batch[0]
        energy = energy_batch[0]

        x_pitch = x
        # x_pitch = x - tf.reduce_mean(x, axis=1, keepdims=True)
        # x_pitch = x_pitch/tf.math.reduce_std(x_pitch, axis=1, keepdims=True)


        # x_energy_reshape = tf.reshape(x, [-1])
        # x_energy = x - tf.reduce_mean(x_energy_reshape)
        # x_energy = x_energy/tf.math.reduce_std(x_energy_reshape)


        y = Reshape(target_shape=(1024, 1, 1), name='input-reshape')(x_pitch)

        for l, f, w, s in zip(layers, filters, widths, strides):
            y = Conv2D(f, (w, 1), strides=s, padding='same',
                       activation='relu', name="conv%d" % l, trainable=False)(y)
            y = BatchNormalization(name="conv%d-BN" % l)(y)
            y = MaxPool2D(pool_size=(2, 1), strides=None, padding='valid',
                          name="conv%d-maxpool" % l, trainable=False)(y)
            y = Dropout(0.25, name="conv%d-dropout" % l)(y)


        y = Permute((2, 1, 3), name="transpose")(y)
        y = Flatten(name="flatten")(y)
        den = Dense(360, activation='sigmoid', name="classifier", trainable=False)
        y = den(y)

        #model = Model(inputs=[x_batch, mask_batch, chroma_batch, energy_batch], outputs=y)
        #model.load_weights('E:\\Vishwaas\\Anaconda3\\envs\\env_tf2\\Lib\\site-packages\\crepe\\model-full.h5', by_name=True)
        # return model
        # pitch_model = Model(inputs=[x, chroma, mask], outputs=y)
        # pitch_model.summary()

        note_emb = den.weights[0]
        note_emb = tf.reduce_mean(tf.reshape(note_emb, [-1, 6, 60]), axis=1)
        note_emb = tf.reduce_mean(tf.reshape(note_emb, [note_dim, -1, 60]), axis=1)
        note_emb = tf.transpose(note_emb, name='note_emb') #60,note_emb


        enc = encoder_2.Encoder(note_emb,sequence_length=sequence_length, N=N,size=note_dim)

        note_emb_red = tf.reduce_mean(tf.reshape(note_emb, [60, int(note_dim/4), 4]), axis=2)

        tonic_chroma = get_tonic_from_cnn(chroma, int(note_dim*3/8), int(note_dim/2), drop_rate) #(60, 32)
        tonic_hist = get_tonic_from_hist(y, int(note_dim*3/8), int(note_dim/2), drop_rate) #(60, 32)
        tonic_sil = get_tonic_from_silence(y, energy, int(note_dim*1/4), drop_rate)  # (60, 32)
        print(tonic_chroma, tonic_hist, tonic_sil)

        tonic_chs = tf.concat([tonic_chroma, tonic_hist, tonic_sil], axis=1)
        tonic_chs_scores_emb = ffnn(tonic_chs, [int(note_dim/2)], drop_rate=drop_rate)
        tonic_chs_scores_emb = Dense(int(note_dim / 4))(tonic_chs_scores_emb)
        tonic_chs_scores = tf.reduce_sum(tf.multiply(note_emb_red, tonic_chs_scores_emb), 1)

        top_k_tonic_scores, top_k_tonics_id = tf.nn.top_k(tonic_chs_scores, tonic_filter, sorted=False)

        # top_k_chs_scores_emb = tf.gather(tonic_chs_scores_emb, top_k_tonics_id)
        # energy_seq = tf.reduce_mean(energy_seq,1)
        # sil_model = Model(inputs=x, outputs=tonic_f2)
        # sil_model = None

        red_y = tf.reshape(y, [-1, 6, 60])
        red_y = tf.reduce_mean(red_y, axis=1)  # (None, 60)
        notes_seq = tf.reshape(red_y, [-1, sequence_length, 60])

        notes_prob = tf.reduce_max(red_y, axis=1)

        notes_n = get_number_of_notes(notes_seq)
        energy_seq = tf.reshape(energy, [-1, sequence_length])
        entropy_seq = get_entropy(red_y, sequence_length)
        mask_seq = tf.reshape(mask, [-1, sequence_length])
        notes_prob_seq = tf.reshape(notes_prob, [-1, sequence_length])

        strength_seq = get_note_strength(energy_seq, entropy_seq, notes_prob_seq)

        tonic_rag_embs = []
        tdms_rag_embs = []

        un_tr_den = Dense(1, activation='relu')
        seq_str_den = (Dense(4, activation='relu'), Dense(1, activation='relu'))

        tdms_conv2d_1 = Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), activation='relu', padding='same', name='tdms_conv2d_1')
        tdms_conv2d_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='valid', name='tdms_conv2d_2')
        tdms_dense = Dense(note_dim, activation='relu', name='tdms_dense')

        for i in range(tonic_filter):

            rolled_notes = tf.roll(red_y, -top_k_tonics_id[i], axis=1) # (None, 60)
            rolled_notes_id = tf.argmax(rolled_notes, axis=1)
            rolled_notes_id_seq = tf.reshape(rolled_notes_id, [-1, sequence_length])

            notes_strength_seq = tf.tile(tf.expand_dims(strength_seq, 2), multiples=[1,1,note_dim])

            un_tr = unlikely_transitions(rolled_notes, sequence_length, un_tr_den, drop_rate)
            seq_str = get_seq_strength(energy_seq, entropy_seq, notes_prob_seq, un_tr, seq_str_den, notes_n, drop_rate)
            tonic_f4 = enc.encode(rolled_notes_id_seq, mask_seq, None, True) #(None, 200, note_dim)
            tonic_f4 = tf.reduce_max(tonic_f4, 1) #(None, note_dim)
            seq_str = tf.tile(tf.expand_dims(seq_str,1), [1,note_dim])
            tonic_f4 = tf.multiply(tonic_f4, seq_str)
            tonic_score = tf.reduce_mean(tonic_f4, axis=0, keepdims=True)  #(1, note_dim)
            if to_train == 'raga':
                tdms_rag_emb = get_tdms(tdms_conv2d_1, tdms_conv2d_2, tdms_dense, rolled_notes, note_dim, i, model_srate, hop_size, drop_rate)
                tdms_rag_embs.append(tdms_rag_emb)
            tonic_rag_embs.append(tonic_score)

        tonic_rag_embs = tf.concat(tonic_rag_embs, axis=0)

        # tonic_rag_embs  = tf.scatter_nd(tf.expand_dims(top_k_tonics_id, 1), tonic_rag_embs, tf.constant([60, 512]))
        # tonic_rag_embs_conc = tf.concat([tonic_rag_embs, tonic_chs_scores_emb], axis=1)
        # tonic_rag_embs_ffnn = ffnn(tonic_rag_embs_conc, [note_dim, note_dim//2], drop_rate=0)
        #
        # tonic_rag_embs_ffnn = tf.squeeze(Dense(1)(tonic_rag_embs_ffnn), axis=1)
        tonic_rag_embs_ffnn = ffnn(tonic_chs_scores_emb, [note_dim // 8], drop_rate=0)
        tonic_rag_embs_ffnn = tf.squeeze(Dense(1)(tonic_rag_embs_ffnn), axis=1)

        tonic_logits = tf.nn.softmax(tonic_rag_embs_ffnn)

        # tonic_logits = tf.scatter_nd(tf.expand_dims(top_k_tonics_id,1), tonic_scores, tf.constant([60]))
        # tonic_logits= tf.add(tonic_logits, (tf.cast(tf.equal(tonic_logits, 0), tf.float32)*1e-9))
        # tonic_logits = tf.identity(tonic_logits, name='tonic')
        # best_tonic = top_k_tonics_id[tf.argmax(tf.concat(tonic_scores, axis=0))]
        tonic_logits = tf.expand_dims(tonic_logits, 0, name='tonic')
        tonic_model = Model(inputs=[x_batch, mask_batch, chroma_batch, energy_batch], outputs=tonic_logits)
        tonic_model.summary()
        tonic_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

        if to_train=='tonic':
            tonic_model.load_weights('model/model-full.h5', by_name=True)
            return tonic_model

        tdms_rag_embs = tf.concat(tdms_rag_embs, axis=0)
        loss_weights = config['loss_weights']
        tdms_rag_embs = tf.concat([tdms_rag_embs, tonic_rag_embs], axis=1)
        tdms_rag_embs_ffnn = ffnn(tdms_rag_embs, [2*note_dim], drop_rate=0)
        tonic_scores_attn = tf.tile(tf.expand_dims(tonic_scores,1), [1, 2*note_dim])
        tonic_scores_attn = tf.nn.softmax(tonic_scores_attn, axis=0)
        tdms_rag = tf.reduce_sum(tf.multiply(tdms_rag_embs_ffnn, tonic_scores_attn), axis=0, keepdims=True)

        rag_logits = Dense(n_labels, activation='softmax', name='raga')(tdms_rag)
        rag_model = Model(inputs=[x_batch, mask_batch, chroma_batch, energy_batch], outputs=[tonic_logits, rag_logits])
        rag_model.compile(loss={'tf_op_layer_tonic': 'binary_crossentropy', 'raga': 'categorical_crossentropy'},
                      optimizer='adam', metrics={'tf_op_layer_tonic': 'binary_accuracy', 'raga': 'accuracy'}, loss_weights={'tf_op_layer_tonic': loss_weights[0], 'raga': loss_weights[1]})

        rag_model.summary()
        # rag_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        rag_model.load_weights('model/model-full.h5', by_name=True)
        return rag_model

        # package_dir = os.path.dirname(os.path.realpath(__file__))
        # filename = "model-{}.h5".format(model_capacity)
        # model.load_weights(os.path.join(package_dir, filename))
        # rag_model = Model(inputs=x, outputs=logits)
        # model.load_weights('E:\\Vishwaas\\Anaconda3\\envs\\env_tf2\\Lib\\site-packages\\crepe\\model-full.h5', by_name=True)

        # w1 = den.weights[0]  #2048,360
        # w1 = tf.reduce_mean(tf.reshape(w1, [-1, 6, 60]), axis=1)
        # w1 = tf.reduce_mean(tf.reshape(w1, [512, 4, 60]), axis=1)
        # # MaxPool1D()
        # # w1 = tf.reduce_mean(tf.reshape(w1, [-1, 12, 5]), axis=2)
        # w1 = tf.transpose(w1)
        # mat = np.zeros([60,60])
        # for i in range(60):
        #     for j in range(60):
        #
        #         a = float(tf.reduce_sum(tf.multiply(w1[i], w1[j])))
        #         mat[i, j] = a
        #
        # plt.imshow(mat, cmap='hot', interpolation='nearest')
        # plt.show()

        # model.compile('adam', 'binary_crossentropy')

        # models[model_capacity] = model
    #
    # return models[model_capacity], tonic_model, rag_model
    # return models[model_capacity], energy_model, tonic_model, sil_model

def get_note_strength(energy_seq, entropy_seq, notes_prob_seq, drop_rate=0.2):
    stacked = tf.stack([energy_seq, entropy_seq, notes_prob_seq],2)
    stacked = Dense(3, activation='relu')(stacked)
    d1 = Dense(1, activation='relu')(stacked)
    return tf.squeeze(d1, 2)

def get_seq_strength(energy_seq, entropy_seq, notes_prob_seq, un_tr, seq_str_den, notes_n, drop_rate=0.2):
    energy_seq = tf.reduce_max(energy_seq, 1, keepdims=True)
    entropy_seq = tf.reduce_max(entropy_seq, 1, keepdims=True)
    notes_prob_seq = tf.reduce_max(notes_prob_seq, 1, keepdims=True)
    notes_n= tf.expand_dims(notes_n, 1)
    stacked = tf.concat([energy_seq, entropy_seq, notes_prob_seq, notes_n, un_tr],1)
    d1= seq_str_den[0](stacked)
    d2 = seq_str_den[1](d1)
    return tf.squeeze(d2, 1)

def get_tdms(tdms_conv2d_1, tdms_conv2d_2, tdms_dense, y, note_dim, j, model_srate=16000, hop_length=160, drop_rate=0.2):
    n_frames = 1 + int((model_srate - 1024) / hop_length)
    # y = tf.math.log(y)
    tdms = []
    for i in range(0,30,1):
        frames_2_shift = int(n_frames*i/10)
        shifted_y = tf.roll(y, -frames_2_shift, axis=0)  #(None, 60)
        y_truc = tf.transpose(y[:-frames_2_shift,:])
        y__shifted_truc = shifted_y[:-frames_2_shift, :]
        tdms.append(tf.matmul(y_truc, y__shifted_truc)) #(60, 60)

    tdms = tf.stack(tdms)  #(30,60,60)
    # tdms = tf.concat([tdms, tf.transpose(tdms, [1,0,2])], axis=2) #(60,60,120)
    tdms = tf.expand_dims(tdms, 0)
    tdms = tf.transpose(tdms, [0,2,3,1])

    z = tdms_conv2d_1(tdms)
    z = BatchNormalization()(z)
    z = MaxPool2D(pool_size=(2, 2), strides=None, padding='valid')(z)
    z = Dropout(drop_rate)(z)
    z = tdms_conv2d_2(z)
    z = MaxPool2D(pool_size=(2, 2), strides=None, padding='valid')(z)
    z = BatchNormalization()(z)
    z = Dropout(drop_rate)(z)
    z = Flatten(name='tdms_flatten_{}'.format(j))(z)
    z = tdms_dense(z)
    # outputs = tf.transpose(tf.squeeze(outputs, 1))
    # outputs = Dense(1, activation='relu')(outputs)
    # z = tf.squeeze(z, 0)
    return z

def get_number_of_notes(notes_seq):
    notes_seq_red = tf.reduce_mean(notes_seq, axis=1)  #(None, 60)
    d1 = Dense(60, activation='relu')(notes_seq_red)
    return tf.reduce_mean(d1, axis=1)


def ffnn(inputs, hidden_size, drop_rate=0.2):
    x = inputs
    for hs in hidden_size:
        den = Dense(hs, activation='relu')(x)
        x = Dropout(drop_rate)(den)
    return x

def unlikely_transitions(rolled_notes, sequence_length, den_weight, drop_rate=0.2):
    # (None, 60)
    rolled_notes_seq = tf.reshape(rolled_notes, [-1, sequence_length, 60])
    rolled_notes_seq = tf.reduce_mean(rolled_notes_seq, 1) #(-1, 60)
    r_R = rolled_notes_seq[:, 5:15]  #(None, 10)
    g_G  =rolled_notes_seq[:, 15:25]
    # m_M = den_weights[2](rolled_notes_seq[:, 25:35]) #try?
    d_D = rolled_notes_seq[:, 40:50]
    n_N = rolled_notes_seq[:, 50:60]

    adj_notes = tf.stack([r_R, g_G, d_D, n_N], 0)  #(4, None, 10)
    adj_notes = tf.transpose(adj_notes, [1,0,2])  #(None, 4,10)
    return 1-tf.squeeze(den_weight(adj_notes),2)

def get_entropy(red_y, sequence_length):
    y = tf.nn.softmax(red_y, axis=1)
    entropy = tf.multiply(y, tf.math.log(y))
    entropy = tf.reduce_sum(entropy, 1)
    entropy_seq = tf.reshape(entropy, [-1, sequence_length])
    return entropy_seq

# @tf.function
def get_tonic_from_silence(y, energy, tonic_emb_size=32, drop_rate=0.2):
    #y, energy, int(note_dim*1/16), drop_rate
    # zarg = tf.constant([1, 1, 5, 3, 5, 2, 2, 7, 6, 6, 11, 3, 3, 3])
    # energy = tf.constant([0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    # z = tf.nn.softmax(y, axis=1) #try this
    z = tf.reshape(y, [-1, 6, 60])
    z = tf.reduce_mean(z, axis=1) #(None, 60)
    # z = tf.reshape(z, [-1,12,5])
    # z = tf.reduce_mean(z, axis=2)  # (None, 12)
    zarg = tf.argmax(z, axis=1)
    energy = (energy - tf.reduce_min(energy))/(tf.reduce_max(energy)- tf.reduce_min(energy))
    energy = tf.cast(tf.math.round(energy), tf.int32)
    # energy = energy - tf.re
    # return energy
    # energy = tf.cast(energy>0.01, tf.int32)
    # energy = tf.cast(tf.round(tf.sigmoid(energy)), tf.int32)
    energy_r = tf.roll(energy, 1, axis=-1)
    energy_r = tf.concat([[0], energy_r[1:]], axis=-1)

    delta_en = energy - energy_r
    delta_en_abs = tf.abs(delta_en)

    delta_en_csum = tf.math.cumsum(delta_en_abs)

    delta_en_csum_pos = tf.squeeze(tf.where(tf.greater(delta_en_csum, 0)), axis=1)

    zarg = tf.gather(zarg, delta_en_csum_pos)
    delta_en_csum = tf.gather(delta_en_csum, delta_en_csum_pos)
    energy = tf.gather(energy, delta_en_csum_pos)
    delta_en = tf.gather(delta_en, delta_en_csum_pos)

    seg_sum_1 = tf.math.segment_sum(1 - energy, delta_en_csum)
    seg_sum_where = tf.squeeze(tf.where(tf.greater(seg_sum_1, 0)), axis=1)
    seg_sum_1 = tf.gather(seg_sum_1, seg_sum_where)

    delta_en_mo = tf.squeeze(tf.where(tf.equal(delta_en, -1)), axis=1) - 1
    zarg_deta = tf.gather(zarg, delta_en_mo)
    seg_sum_usm = tf.math.unsorted_segment_mean(seg_sum_1, zarg_deta, 60)

    logspace_idx = tf.cast(tf.floor(tf.math.log1p(seg_sum_usm) / math.log(2)), tf.int32) + 3
    use_identity = tf.cast(seg_sum_usm <= 4, tf.int32)
    seg_sum_usm = tf.cast(seg_sum_usm, tf.int32)
    combined_idx = use_identity * seg_sum_usm + (1 - use_identity) * logspace_idx

    clipped = tf.clip_by_value(combined_idx, 0, 9)

    emb = tf.keras.layers.Embedding(10, tonic_emb_size, input_length=60)
    return emb(clipped)
    # emb = tf.Variable(initializer([10, 128]), dtype=tf.float32)
    # return tf.gather(emb, clipped)
    # return clipped

# @tf.function
def get_tonic_from_hist(y, tonic_emb_size=32, tonic_cnn_filters=128, drop_rate=0.2):
    #y, int(note_dim*3/32), int(note_dim/2), drop_rate
    #z = tf.nn.softmax(y, axis=1) #try this
    z = tf.reduce_mean(y, axis=0)
    z = tf.reshape(z, [6,60])
    z = tf.reduce_mean(z, axis=0)
    kernel_size = [5, 10, 15, 20]

    outputs = []
    for ks in kernel_size:
        bz = tf.concat([z, z[:ks]], axis=0)
        bz = tf.reshape(bz, [1, -1, 1]) #
        conv = Conv1D(filters=tonic_cnn_filters, kernel_size=ks, strides=1, activation='relu', padding='valid')(bz) ##((60+ks)/ks, ks, 1)
        conv = tf.squeeze(conv, axis=0)
        conv = conv[:-1, :]
        conv = Dropout(drop_rate)(conv)
        conv = Dense(tonic_emb_size, activation='relu')(conv)
        outputs.append(conv)

    outputs = tf.concat(outputs, 1)
    outputs = ffnn(outputs, [tonic_emb_size], drop_rate=drop_rate)
    return outputs

# @tf.function
def get_tonic_from_cnn(chroma, tonic_emb_size=32, tonic_cnn_filters=128, drop_rate=0.1):

    #chroma, int(note_dim*3/32), int(note_dim/2), drop_rate
    chroma = tf.expand_dims(chroma,0)
    chroma = tf.expand_dims(chroma, 3)
    y = Conv2D(tonic_cnn_filters, (5, 5), strides=1, padding='same', activation='relu', name='tonic_cnn_1')(chroma)
    y = Conv2D(tonic_cnn_filters, (5, 5), strides=1, padding='same', activation='relu', name='tonic_cnn_2')(y)
    y = Conv2D(tonic_cnn_filters, (5, 5), strides=1, padding='same', activation='relu', name='tonic_cnn_3')(y)
    y = Dropout(drop_rate)(y)
    y = Conv2D(tonic_cnn_filters, (5, 5), strides=1, padding='same', activation='relu', name='tonic_cnn_4')(y)
    y = Conv2D(tonic_cnn_filters, (5, 5), strides=1, padding='same', activation='relu', name='tonic_cnn_5')(y) #(1, 60, -1, 128)
    y = Dropout(drop_rate)(y)
    y = tf.squeeze(y, 0)
    y = tf.reduce_mean(y, 1)
    y = Dense(tonic_emb_size, activation='relu', name = 'tonic_cnn_dense_1')(y)
    return y #(60,32)



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

def train():
    task = 'tonic'
    config = pyhocon.ConfigFactory.parse_file("crepe/experiments.conf")[task]
    training_generator = DataGenerator(task, 'hindustani', 'train', config)
    validation_generator = DataGenerator(task, 'hindustani', 'validate', config)
    model = build_and_load_model(config, task)
    # model.fit(generator)
    # model.fit(x=training_generator,
    #                     validation_data=validation_generator, verbose=2, epochs=15, shuffle=True, batch_size=1)
    checkpoint = ModelCheckpoint("model/tonic_model.hdf5", monitor='loss', verbose=1,
                                 save_best_only=True, mode='auto', period=1)
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator, verbose=1, epochs=15, shuffle=True, callbacks=[checkpoint])

def test():
    task = 'tonic'
    config = pyhocon.ConfigFactory.parse_file("crepe/experiments.conf")[task]
    test_generator = DataGenerator(task, 'hindustani', 'test', config)
    model = build_and_load_model(config, task)
    # model.fit(generator)
    # model.fit(x=training_generator,
    #                     validation_data=validation_generator, verbose=2, epochs=15, shuffle=True, batch_size=1)
    model.load_weights('model/tonic_model.hdf5', by_name=True)
    print(model.predict(x=test_generator))

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

