import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
import math

class Encoder:

    def __init__(self, note_emb, enc_num=1, sequence_length=1000, size=2048, drop_rate=0.1, N=6):
        ones_init = tf.keras.initializers.ones()
        zeros_init = tf.keras.initializers.zeros()
        initializer = tf.initializers.GlorotUniform()
        self.size = size
        if enc_num==1:
            self.dense_norm = [Dense(units=size, kernel_initializer=ones_init, bias_initializer=zeros_init, name='encoder_dense_norm_{}'.format(i)) for i in range(2*N+1)]
            self.dense_lin = [Dense(size, kernel_initializer=initializer, bias_initializer=initializer, name='encoder_dense_lin_{}'.format(i)) for i in range(4*N)]
            # self.dense_lin = [Dense(size) for _ in range(4*N)]
            self.dense_ff1 = [Dense(size*4, kernel_initializer=initializer, bias_initializer=initializer, name='encoder_dense_ff1_{}'.format(i)) for i in range(N)]
            self.dense_ff2 = [Dense(size, kernel_initializer=initializer, bias_initializer=initializer, name='encoder_dense_ff2_{}'.format(i)) for i in range(N)]
            self.rel_pos_emb = [
                tf.Variable(initializer([60, size]), dtype=tf.float32, name='encoder_rel_pos_emb_{}'.format(i)) for i in
                range(2 * N)]
        else:
            self.dense_norm = [Dense(units=size, kernel_initializer=ones_init, bias_initializer=zeros_init, name='encoder_2_dense_norm_{}'.format(i)) for i in range(2*N+1)]
            self.dense_lin = [Dense(size, kernel_initializer=initializer, bias_initializer=initializer, name='encoder_2_dense_lin_{}'.format(i)) for i in range(4*N)]
            # self.dense_lin = [Dense(size) for _ in range(4*N)]
            self.dense_ff1 = [Dense(size*4, kernel_initializer=initializer, bias_initializer=initializer, name='encoder_2_dense_ff1_{}'.format(i)) for i in range(N)]
            self.dense_ff2 = [Dense(size, kernel_initializer=initializer, bias_initializer=initializer, name='encoder_2_dense_ff2_{}'.format(i)) for i in range(N)]
            self.rel_pos_emb = [
                tf.Variable(initializer([60, size]), dtype=tf.float32, name='encoder_2_rel_pos_emb_{}'.format(i)) for i in
                range(2 * N)]


        # sess = tf.compat.v1.Session()
        # init = tf.compat.v1.initialize_all_variables()
        # sess.run(init)

        self.dropout = Dropout(rate=drop_rate)
        self.pe = self.positional_emb(size)
        self.note_emb = note_emb
        self.N = N
        self.sequence_length = sequence_length

        self.dense_norm_counter = 0
        self.dense_lin_counter = 0
        self.dense_ff1_counter = 0
        self.dense_ff2_counter = 0
        self.rel_pos_emb_counter= 0

    def positional_emb(self, d_model, max_len=5000):
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

        return pe

    def _relative_attention_inner(self, x, y, z, transpose):
        batch_size = tf.shape(x)[0]
        heads = 8
        length = tf.shape(x)[2]

        # xy_matmul is [batch_size, heads, length or 1, length or depth]
        xy_matmul = tf.matmul(x, y, transpose_b=transpose)
        # x_t is [length or 1, batch_size, heads, length or depth]
        x_t = tf.transpose(x, [2, 0, 1, 3])
        # x_t_r is [length or 1, batch_size * heads, length or depth]
        x_t_r = tf.reshape(x_t, [length, heads * batch_size, -1])
        # x_tz_matmul is [length or 1, batch_size * heads, length or depth]
        x_tz_matmul = tf.matmul(x_t_r, z, transpose_b=transpose)
        # x_tz_matmul_r is [length or 1, batch_size, heads, length or depth]
        x_tz_matmul_r = tf.reshape(x_tz_matmul, [length, batch_size, heads, -1])
        # x_tz_matmul_r_t is [batch_size, heads, length or 1, length or depth]
        x_tz_matmul_r_t = tf.transpose(x_tz_matmul_r, [1, 2, 0, 3])
        return xy_matmul + x_tz_matmul_r_t

    def _generate_relative_positions_matrix(self, length, max_relative_position):
        range_vec = tf.range(length)
        range_mat = tf.reshape(tf.tile(range_vec, [length]), [length, length])
        distance_mat = range_mat - tf.transpose(range_mat)
        distance_mat_clipped = tf.clip_by_value(distance_mat, -max_relative_position,
                                                max_relative_position)
        # Shift values to be >= 0. Each integer still uniquely identifies a relative
        # position difference.
        final_mat = distance_mat_clipped + max_relative_position
        return final_mat

    def _generate_relative_positions_embeddings(self, length, depth,
                                                max_relative_position):
        relative_positions_matrix = self._generate_relative_positions_matrix(
            length, max_relative_position)
        vocab_size = max_relative_position * 2 + 1

        # initializer = tf.contrib.layers.xavier_initializer()
        # lut = tf.Variable(initializer([vocab_size, depth]))  # (V, d_model)
        # Generates embedding for each relative position of dimension depth.
        #     embeddings_table = tf.get_variable("embeddings", [vocab_size, depth])
        lut = self.rel_pos_emb[self.rel_pos_emb_counter]
        self.rel_pos_emb_counter+=1
        embeddings = tf.gather(lut, relative_positions_matrix)
        return embeddings

    def dot_product_attention_relative(self, q, k, v, max_relative_position, mask=None, dropout_rate=0.1):
        depth = k.get_shape().as_list()[3]
        # length = tf.shape(k)[2]
        length = self.sequence_length
        if mask is not None:
            mask = tf.expand_dims(mask, 1)  # encoder: (b,1,1, V-1); decoder: (b,1,V-2, V-2)
        relations_keys = self._generate_relative_positions_embeddings(length, depth, max_relative_position)
        relations_values = self._generate_relative_positions_embeddings(length, depth, max_relative_position)

        logits = self._relative_attention_inner(q, k, relations_keys, True)

        if mask is not None:
            mask_dim1 = tf.shape(mask)[1]  # encoder: (1); decoder: (1)
            mask_dim2 = tf.shape(mask)[2]  # encoder: (1); decoder: (V-2)

            scores_dim1 = tf.shape(logits)[1]  # (h)
            scores_dim2 = tf.shape(logits)[2]  # encoder: V-1; decoder: V-2
            mask = tf.tile(mask,
                           [1, tf.cast(scores_dim1 / mask_dim1, tf.int32), tf.cast(scores_dim2 / mask_dim2, tf.int32),
                            1])  # mask.shape = scores.shape
            mask = tf.cast(mask, tf.float32)
            logits = tf.multiply(tf.cast(tf.not_equal(mask, 0), tf.float32), logits) + -1e9 * tf.cast(
                tf.equal(mask, 0), tf.float32)  # put 1e-9 whereever mask=0

        weights = tf.nn.softmax(logits)
        weights = tf.nn.dropout(weights, dropout_rate)
        return self._relative_attention_inner(weights, v, relations_values, False)

    def embeddings(self, x, rolled_notes_prob):
        emb = tf.gather(self.note_emb, x)
        # emb = self.note_emb(x)
        if rolled_notes_prob is not None:
            emb = tf.multiply(emb, rolled_notes_prob)
        emb2 = emb + tf.transpose(self.pe[:self.sequence_length], [1, 0, 2])
        return self.dropout(emb2)

    def encode(self, x, mask, rolled_notes_prob=None, use_emb = True):
        # x: (b, V-1, d_model);
        # mask: (b,1, V-1);
        # z = tf.expand_dims(x[a],0)


        if use_emb:
            emb = self.embeddings(x, rolled_notes_prob)
        else:
            emb = x
        # src_mask = tf.expand_dims(tf.expand_dims(mask[a], axis=0), axis=0)
        src_mask = tf.expand_dims(mask,1)
        for i in range(self.N):
            emb = self.encoder_layer(emb, src_mask)
        toreturn =  self.layer_norm(emb)
        self.reset_counts()
        return toreturn

    def encoder_layer(self, x, mask):
        # x: (b, V-1, d_model);
        # mask: (b,1, V-1);
        norm1 = self.layer_norm(x)

        z, attn = self.multi_head_attn(norm1, norm1, norm1, mask, h=8)
        slc1 = x + self.dropout(z)
        # slc1 = x + tf.nn.dropout(z, rate=drop_rate)
        norm2 = self.layer_norm(slc1)
        ff = self.feed_forward(norm2)
        # slc2 = slc1 + tf.nn.dropout(ff, rate=drop_rate)
        slc2 = slc1 + self.dropout(ff)
        return slc2

    ones_init = tf.keras.initializers.ones()
    zeros_init = tf.keras.initializers.zeros()

    def layer_norm(self, x, eps=1e-6):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        std = tf.math.reduce_std(x, axis=-1, keepdims=True)
        z = (x - mean) / (std + eps)

        print('self.dense_norm_counter', self.dense_norm_counter)
        den  = self.dense_norm[self.dense_norm_counter](z)
        self.dense_norm_counter+=1
        return den

    def attention(self, query, key, value, mask):
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
            mask = tf.tile(mask,
                           [1, tf.cast(scores_dim1 / mask_dim1, tf.int32), tf.cast(scores_dim2 / mask_dim2, tf.int32),
                            1])  # mask.shape = scores.shape
            mask = tf.cast(mask, tf.float32)
            scores = tf.multiply(tf.cast(tf.not_equal(mask, 0), tf.float32), scores) + -1e9 * tf.cast(
                tf.equal(mask, 0), tf.float32)  # put 1e-9 whereever mask=0
        # p_attn = tf.nn.softmax(scores, axis=-1)  # scores.shape
        p_attn = tf.keras.layers.Softmax(axis=-1)(scores)
        # p_attn = tf.compat.v1.layers.dropout(p_attn, rate=drop_rate)  # scores.shape
        # p_attn = tf.nn.dropout(p_attn, rate=drop_rate)
        p_attn = self.dropout(p_attn)
        return tf.matmul(p_attn,
                         value), p_attn  # (encoder: (b, h, V-1, d_model/h); decoder: (b, h, V-2, d_model/h)) , scores.shape

    def multi_head_attn(self, q, k, v, mask, h=8):
        # k,v,q: (b, V-1, d_model)
        # size= d_model
        # if decoder: V-1 = V-2

        query = self.linear(q)  # (b, V-1, d_model)
        key = self.linear(k)  # (b, V-1, d_model)
        value = self.linear(v)  # (b, V-1, d_model)
        d_k = self.size // h
        bs = tf.shape(query)[0]
        query = tf.reshape(query, [bs, h, self.sequence_length, d_k])  # (b, h, V-1, d_model/h)
        key = tf.reshape(key, [bs, h, self.sequence_length, d_k])  # (b, h, V-1, d_model/h)
        value = tf.reshape(value, [bs, h, self.sequence_length, d_k])  # (b, h, V-1, d_model/h)
        x, attn = self.attention(query, key, value, mask)
        # x = self.dot_product_attention_relative(query, key, value, 5, mask)
        x = tf.reshape(x, [bs, self.sequence_length, h * d_k])
        return self.linear(x), None

    def linear(self, x):
        # x: (None, dim_1)
        den  = self.dense_lin[self.dense_lin_counter](x)
        # den = Dense(2048)(x)
        self.dense_lin_counter+=1
        return den

    def linear_ff1(self, x):
        # x: (None, dim_1)
        den  = self.dense_ff1[self.dense_ff1_counter](x)
        self.dense_ff1_counter+=1
        return den

    def linear_ff2(self, x):
        # x: (None, dim_1)
        den  = self.dense_ff2[self.dense_ff2_counter](x)
        self.dense_ff2_counter+=1
        return den

    def feed_forward(self, x):
        l1 = self.linear_ff1(x)
        re = tf.nn.relu(l1)
        dr = self.dropout(re)
        l2 = self.linear_ff2(dr)
        return l2

    def reset_counts(self):
        self.dense_lin_counter=0
        self.dense_ff1_counter=0
        self.dense_ff2_counter=0
        self.dense_norm_counter=0
        self.rel_pos_emb_counter=0