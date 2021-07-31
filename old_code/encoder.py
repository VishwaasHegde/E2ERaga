import tensorflow as tf
import math

V = 60
d_model = 2048
batch_size = 1

def encode(x, note_emb, mask=None, size=d_model, drop_rate=0.1, N=6):
    # z = tf.argmax(x, axis=1)
    # z = tf.reduce_mean(z, axis=1) #(None, 60)
    # emb = tf.gather(note_emb, x)
    return encoder(note_emb, mask, size, drop_rate, N)

def encoder(x, mask, size, drop_rate, N=6):
    #x: (b, V-1, d_model);
    #mask: (b,1, V-1);
    for i in range(N):
        x = encoder_layer(x, mask, size, drop_rate)
    return layer_norm(x, size)


def encoder_layer(x, mask, size, drop_rate):
    # x: (b, V-1, d_model);
    # mask: (b,1, V-1);
    norm1 = layer_norm(x, size)

    z, attn = multi_head_attn(norm1, norm1, norm1, mask, size, drop_rate, h=8)
    slc1 = x + tf.keras.layers.Dropout(rate = drop_rate)(z)
    # slc1 = x + tf.nn.dropout(z, rate=drop_rate)
    norm2 = layer_norm(slc1, size)
    ff = feed_forward(norm2, drop_rate, size)
    # slc2 = slc1 + tf.nn.dropout(ff, rate=drop_rate)
    slc2 = slc1 + tf.keras.layers.Dropout(rate = drop_rate)(z)
    return slc2

ones_init = tf.keras.initializers.ones()
zeros_init = tf.keras.initializers.zeros()

def layer_norm(x, size, eps=1e-6):
    mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    std = tf.math.reduce_std(x, axis=-1, keepdims=True)
    z = (x - mean) / (std + eps)

    return tf.keras.layers.Dense(units=size, kernel_initializer=ones_init, bias_initializer=zeros_init)(z)
    # return tf.compat.v1.layers.dense(z, units=size, kernel_initializer='ones', bias_initializer='zeros')


def attention(query, key, value, drop_rate, mask=None):
    "Compute 'Scaled Dot Product Attention'"
    # query, key, value: (b, h, V-1, d_model/h)
    if mask is not None:
        mask = tf.expand_dims(mask, 1)  # encoder: (b,1,1, V-1); decoder: (b,1,V-2, V-2)
    d_k = tf.cast(tf.shape(query)[-1], tf.float32)
    scores = tf.divide(tf.matmul(query, tf.transpose(key, [0, 1, 3, 2])),
                       tf.sqrt(d_k))  # encoder: (b,h,V-1,V-1); decoder: (b,h,V-2,V-2)

    if mask is not None:
        mask_dim1 = tf.shape(mask)[1]  # encoder: (1); decoder: (1)
        mask_dim2 = tf.shape(mask)[2]  # encoder: (1); decoder: (V-2)

        scores_dim1 = tf.shape(scores)[1]  # (h)
        scores_dim2 = tf.shape(scores)[2]  # encoder: V-1; decoder: V-2
        mask = tf.tile(mask, [1, tf.cast(scores_dim1 / mask_dim1, tf.int32), tf.cast(scores_dim2 / mask_dim2, tf.int32),
                                          1])  # mask.shape = scores.shape
        mask = tf.cast(mask, tf.float32)
        scores = tf.multiply(tf.cast(tf.not_equal(mask, 0), tf.float32), scores) + -1e9 * tf.cast(
            tf.equal(mask, 0),tf.float32)  # put 1e-9 whereever mask=0
    # p_attn = tf.nn.softmax(scores, axis=-1)  # scores.shape
    p_attn = tf.keras.layers.Softmax(axis=-1)(scores)
    # p_attn = tf.compat.v1.layers.dropout(p_attn, rate=drop_rate)  # scores.shape
    # p_attn = tf.nn.dropout(p_attn, rate=drop_rate)
    p_attn = tf.keras.layers.Dropout(rate=drop_rate)(p_attn)
    return tf.matmul(p_attn,
                     value), p_attn  # (encoder: (b, h, V-1, d_model/h); decoder: (b, h, V-2, d_model/h)) , scores.shape


def multi_head_attn(q, k, v, mask, size, drop_rate, h=8):
    # k,v,q: (b, V-1, d_model)
    # size= d_model
    # if decoder: V-1 = V-2

    query = linear(q, size, size, True)  # (b, V-1, d_model)
    key = linear(k, size, size, True)  # (b, V-1, d_model)
    value = linear(v, size, size, True)  # (b, V-1, d_model)
    d_k = size // h
    bs = tf.shape(query)[0]
    query = tf.reshape(query, [bs, h, -1, d_k])  # (b, h, V-1, d_model/h)
    key = tf.reshape(key, [bs, h, -1, d_k])  # (b, h, V-1, d_model/h)
    value = tf.reshape(value, [bs, h, -1, d_k])  # (b, h, V-1, d_model/h)
    x, attn = attention(query, key, value, drop_rate, mask)
    x = tf.reshape(x, [bs, -1, h * d_k])
    return linear(x, d_model, d_model, True), attn

def linear(x, dim_1, dim_2, is_3d=True):
    #x: (None, dim_1)
    initializer = tf.initializers.GlorotUniform()

    return tf.keras.layers.Dense(dim_2, kernel_initializer=initializer, bias_initializer=initializer)(x)
    # if is_3d:
    #     w = tf.Variable(initializer([1,dim_1, dim_2]))
    #     b = tf.Variable(initializer([1, 1, dim_2]))
    #     f = tf.cast(tf.shape(x)[0], tf.int32)
    #     w = tf.tile(w, [f,1,1])
    #     b = tf.tile(b, [f,1,1])
    #     return tf.matmul(x, w)+b
    # else:
    #     w = tf.Variable(initializer([dim_1, dim_2]))
    #     b = tf.Variable(initializer([1, dim_2]))
    #     return tf.matmul(x, w)+b

def feed_forward(x, drop_rate, size, d_ff=2048*2):
    l1 = linear(x, size, d_ff, True)
    re = tf.nn.relu(l1)
    dr = tf.compat.v1.layers.dropout(re, rate=drop_rate)
    l2 = linear(dr, d_ff, size, True)
    return l2

def embeddings(x, vocab, size):
    #x: (b, V-1)
    initializer =  tf.initializers.GlorotUniform()
    lut = tf.Variable(initializer([vocab, size])) #(V, d_model)
    return tf.gather(lut, x) * math.sqrt(size) #(b, V-1, d_model)


def positional_emb(x, d_model, drop_rate, max_len=5000):
    position = tf.cast(tf.range(max_len), tf.float32)
    position = tf.expand_dims(position, 1)
    div_term = tf.exp(tf.multiply(tf.cast(tf.range(0, d_model, 2), tf.float32), -(math.log(10000.0) / d_model)))
    div_term = tf.expand_dims(div_term, 0)
    even = tf.transpose(tf.sin(position * div_term))  # (d_model/2,max_len)
    odd = tf.transpose(tf.cos(position * div_term))
    even_odd = tf.transpose(tf.stack([even, odd], axis=2), [0, 2, 1])

    ind_1 = tf.range(tf.cast(d_model / 2, tf.int32))
    ind_1 = tf.stack([ind_1, ind_1], axis=1)
    ind_1 = tf.reshape(ind_1, [-1])
    ind_2 = tf.tile([0, 1, 0, 1], [tf.cast(d_model / 4, tf.int32)])
    ind_12 = tf.stack([ind_1, ind_2], axis=1)
    pe = tf.expand_dims(tf.transpose(tf.gather_nd(even_odd, ind_12)), axis=0)
    pe = tf.transpose(pe, [1, 0, 2])
    x = x + tf.transpose(tf.gather(pe, tf.range(tf.shape(x)[1])), [1, 0, 2])
    return tf.nn.dropout(x, rate=drop_rate)

# def sequential(x, vocab, size, drop_rate):
#     emb = embeddings(x, vocab, size)
# #     return emb
#     return positional_emb(emb, size, drop_rate)

def sequential(x, emb, size, drop_rate):
#     return emb
    pe =  positional_emb(emb, size, drop_rate)
    x = x + tf.transpose(tf.gather(pe, tf.range(tf.shape(x)[1])), [1, 0, 2])
    return tf.nn.dropout(x, rate=drop_rate)

# import numpy as np
#
# bsrc = tf.constant(np.random.randn(2,5,512), dtype=tf.float32)
#
# src_emb = sequential(bsrc, d_model, 0.1)
#
# enc = encoder(bsrc, mask=None, size=512, drop_rate=0.1, N=2)
# print(enc)


