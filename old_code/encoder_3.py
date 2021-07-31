import tensorflow as tf
import numpy as np
import math
from tensorflow.keras.layers import Input, Reshape, Conv2D, BatchNormalization, Softmax, Conv1D, Bidirectional
from tensorflow.keras.layers import MaxPool2D, Dropout, Permute, Flatten, Dense, MaxPool1D, AvgPool1D

count=0
def encode(inputs, note_emb, note_strength, inputs_mask, n_layers, heads=4, dropout=0.2, d_model=128):
    input_embeddings = get_embeddings(inputs, note_emb, note_strength)
    positional_encodings = generate_positional_encodings(d_model)
    input_embeddings = prepare_embeddings(input_embeddings,
                                          positional_encodings=positional_encodings,
                                          dropout=dropout,
                                          is_input=True)

    d_ff = 4*d_model
    encoding = encoder(input_embeddings, mask=inputs_mask, n_layers=n_layers, heads=heads,
                       dropout=dropout, d_ff=d_ff)
    return encoding

def get_embeddings(input_ids: tf.Tensor, note_emb, note_strength):
    emb = tf.gather(note_emb, input_ids)
    if note_strength is not None:
        emb = tf.multiply(emb, tf.expand_dims(note_strength,2))

    return emb

def generate_positional_encodings(d_model: int, max_len: int = 5000):
    encodings = np.zeros((max_len, d_model), dtype=float)
    position = np.arange(0, max_len).reshape((max_len, 1))
    two_i = np.arange(0, d_model, 2)
    div_term = np.exp(-math.log(10000.0) * two_i / d_model)
    encodings[:, 0::2] = np.sin(position * div_term)
    encodings[:, 1::2] = np.cos(position * div_term)
    return tf.constant(encodings.reshape((1, max_len, d_model)),
                       dtype=tf.float32, name="positional_encodings")

def prepare_embeddings(x: tf.Tensor, *, positional_encodings: tf.Tensor, dropout: float, is_input: bool):
    # _, seq_len, _ = x.shape
    seq_len = tf.shape(x)[1]
    x = x + positional_encodings[:, :seq_len, :]
    x = Dropout(dropout)(x)
    return layer_norm(x)


def get_mean_std(x: tf.Tensor):
#
    mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    squared = tf.square(x - mean)
    variance = tf.reduce_mean(squared, axis=-1, keepdims=True)
    std = tf.sqrt(variance)

    return mean, std

def layer_norm(layer: tf.Tensor):
    global count
    ones_init = tf.keras.initializers.ones()
    zeros_init = tf.keras.initializers.zeros()
    norm_den = Dense(units=layer.shape[-1], kernel_initializer=ones_init, bias_initializer=zeros_init, name='enc_dense_{}'.format(count))
    count+=1
    mean, std = get_mean_std(layer)
    norm = (layer - mean) / (std + 1e-6)
    return norm_den(norm)

def encoder(x: tf.Tensor, *,
            mask: tf.Tensor,
            n_layers: int,
            heads: int, dropout: float, d_ff: int):
    for i in range(n_layers):
        x = encoder_layer(x, mask=mask, index=i, heads=heads, dropout=dropout, d_ff=d_ff)

    return x

def encoder_layer(x: tf.Tensor, *,
                  mask: tf.Tensor, index: int, heads: int,
                  dropout: float, d_ff: int):
    d_model = x.shape[-1]
    attention_out = multi_head_attention(x, x, x,
                                             mask=mask, heads=heads, dropout=dropout)
    added = x + Dropout(dropout)(attention_out)
    x = layer_norm(added)
    ff_out = feed_forward(x, d_model, d_ff, dropout)
    added = x + Dropout(dropout)(ff_out)
    return layer_norm(added)

def feed_forward(x: tf.Tensor,
                 d_model: int, d_ff: int, keep_prob: float):
    global count
    hidden = Dense(d_ff, name='enc_dense_{}'.format(count))(x)
    count+=1
    hidden = tf.nn.relu(hidden)
    hidden = Dropout(keep_prob)(hidden)
    toreturn  = Dense(units=d_model, name='enc_dense_{}'.format(count))(hidden)
    count += 1
    return toreturn

def multi_head_attention(query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, *,
                         mask: tf.Tensor,
                         heads: int,
                         dropout: float):
    global count
    n_batches, seq_len, d_model = query.shape
    n_batches = tf.shape(query)[0]
    seq_len = tf.shape(query)[1]
    query = prepare_for_multi_head_attention(query, heads, "query")
    key = prepare_for_multi_head_attention(key, heads, "key")
    value = prepare_for_multi_head_attention(value, heads, "value")
    mask = tf.expand_dims(mask, axis=1)
    out, _ = attention(query, key, value, mask=mask, dropout=dropout)
    out = tf.transpose(out, perm=[0, 2, 1, 3])
    out = tf.reshape(out, shape=[n_batches, seq_len, d_model])
    # out = tf.reshape(out, shape=[n_batches, -1, d_model])
    toreturn = Dense(units=d_model, name='enc_dense_{}'.format(count))(out)
    count += 1
    return toreturn

def prepare_for_multi_head_attention(x: tf.Tensor, heads: int, name: str):
    global count
    # n_batches, seq_len, d_model = x.shape
    n_batches, seq_len, d_model = tf.shape(x)
    n_batches = tf.shape(x)[0]
    seq_len = tf.shape(x)[1]
    d_model = x.shape[2]
    assert d_model % heads == 0
    d_k = d_model // heads
    x = Dense(units=d_model, name='enc_dense_{}'.format(count))(x)
    count += 1
    x = tf.reshape(x, shape=[n_batches, seq_len, heads, d_k])
    # x = tf.reshape(x, shape=[n_batches, -1, heads, d_k])
    x = tf.transpose(x, perm=[0, 2, 1, 3])
    return x

def attention(query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, *,
              mask: tf.Tensor,
              dropout: float):
    d_k = query.shape[-1]
    scores = tf.matmul(query, tf.transpose(key, perm=[0, 1, 3, 2]))
    scores = scores / tf.constant(math.sqrt(d_k))
    mask_add = ((scores * 0) - 1e9) * (tf.constant(1.) - mask)
    scores = scores * mask + mask_add
    attn = tf.nn.softmax(scores, axis=-1)
    attn = Dropout(dropout)(attn)
    return tf.matmul(attn, value), attn