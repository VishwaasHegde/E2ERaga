import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
from scipy.io import wavfile
from numpy.lib.stride_tricks import as_strided
import librosa
import librosa.display
import pandas as pd
import math
from pydub import AudioSegment
from resampy import resample
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Lambda
import tensorflow as tf
import h5py

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, task, tradition, process, config, random=False, shuffle=True, full_length=False):
        'Initialization'
        self.batch_size = 1
        self.model_srate = config['model_srate']
        self.step_size = config['hop_size']
        self.random = random
        hop_length = int(self.model_srate * self.step_size)
        self.task = task
        self.n_labels = config['n_labels']
        self.cutoff = config['cutoff']
        self.tradition = tradition
        self.sequence_length = int((config['sequence_length'] * self.model_srate - 1024) / hop_length) + 1
        data_path = config[tradition+'_'+process]

        data = pd.read_csv(data_path, sep='\t')
        # data = data.iloc[15:16,:]
        def pitch_path(path):
            pp = path.replace('audio', 'pitches')
            pp = pp[:pp.index('.wav')] + '.pitch'
            return pp

        if full_length:
            data['pitch_path'] = data['path'].apply(pitch_path)
            data['id']= np.arange(data.shape[0])
            data['slice'] = 0
            random = False
        else:
            data = self.get_split_data(data, self.cutoff)

        # data = data.set_index('mbid')
        self.list_IDs = data['path']
        self.data = data
        self.shuffle = shuffle
        self.random = random
        self.full_length = full_length

        bce = tf.keras.losses.BinaryCrossentropy()
        self.bce_layer = Lambda(lambda tensors: bce(tensors[0], tensors[1]))

        self.lm_file = h5py.File('data/RagaDataset/Hindustani/cqt_cache.hdf5', "r")

        # self.indexes = np.arange(len(self.list_IDs))
        self.on_epoch_end()

    # def get_split_data(self, data, cutoff):
    #
    #     all_data = []
    #     id = 0
    #     for path, file_len, tonic, label in data[['path', 'len', 'tonic_fine', 'labels']].values:
    #         if label<5:
    #             n_cutoff = int(file_len/cutoff)
    #             for i in range(n_cutoff):
    #                 json_data = {}
    #                 json_data['id'] = id
    #                 json_data['path'] = path
    #                 json_data['slice'] = i
    #                 json_data['tonic'] = tonic
    #                 json_data['labels'] = label
    #                 all_data.append(json_data)
    #                 id+=1
    #     df = pd.DataFrame(all_data)
    #     df = df.set_index('id')
    #     return df

    def get_split_data(self, data, cutoff):

        all_data = []
        id = 0
        for path, old_path, file_len, tonic, mbid, label in data[['path', 'old_path', 'len', 'tonic_fine', 'mbid', 'labels']].values:
            if label<30:
                n_cutoff = int(file_len/cutoff)
                # n_cutoff = 1000
                # cutoffs = np.random.randint(0,n_cutoff-1,1000)
                for i in range(n_cutoff):
                    json_data = {}
                    json_data['id'] = id
                    json_data['path'] = path
                    # json_data['pitch_path'] = old_path+'.pitch'
                    pitch_path = path.replace('audio', 'pitches')
                    pitch_path = pitch_path[:pitch_path.index('.wav')]+'.pitch'
                    json_data['pitch_path'] = pitch_path
                    json_data['slice'] = i
                    json_data['tonic'] = tonic
                    json_data['mbid'] = mbid
                    json_data['labels'] = label
                    json_data['n_cutoff'] = 1000
                    all_data.append(json_data)
                    id+=1
        df = pd.DataFrame(all_data)
        df = df.set_index('id')
        return df

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index_un):
        'Generate one batch of data'
        # Generate indexes of the batch

        index = self.indexes[index_un]

        if self.task == 'pitch':
            # id = self.data.index[index]
            path = self.data.loc[index, 'path']
            slice_ind = None
            if not self.shuffle:
                slice_ind = self.data.loc[index, 'slice']
            # print(index_un,index, slice_ind)
            X, y = self.__data_generation_pitch(index, path, slice_ind)
            y = self.freq_to_cents(y)
            return X, y

        if self.task=='tonic':
            path = self.data.loc[index, 'path']
            # print(path)
            slice_ind = None
            if not self.random:
                slice_ind = self.data.loc[index, 'slice']
            tonic = self.data.loc[index, 'tonic']
            y_tonic = self.freq_to_cents(tonic)
            y_tonic = np.reshape(y_tonic, [-1, 6, 60])
            y_tonic = np.sum(y_tonic, axis=1)

            # y_tonic = np.array([y_tonic])
            pitches = self.__data_generation_raga(index, path, slice_ind)
            # X, pitches = self.__data_generation_pitch(index, path, slice_ind)
            # X = [X]
            pitches = np.array([pitches])

            # label_freq = self.data.iloc[index, 2]
            # print('label_freq', label_freq)
            # label_freq = self.freq_to_cents(label_freq)
            # label_freq = np.reshape(label_freq, [-1, 6, 60])
            # y = np.sum(label_freq, axis=1)
            transpose_by = np.random.randint(60, dtype=np.int32)
            transpose_by = 0
            y_tonic = np.roll(y_tonic, -transpose_by, axis=1)
            transpose_by = np.array([transpose_by])

            return {'pitches_input': pitches, 'transpose_input': transpose_by}, {'tonic': y_tonic}
        elif self.task=='raga':
            # index = 15
            path = self.data.loc[index, 'path']
            mbid = self.data.loc[index, 'mbid']
            # path = 'data\\RagaDataset\\audio\\077d538e-545a-4098-8997-76742fb26d39.wav'
            # mbid = '077d538e-545a-4098-8997-76742fb26d39.wav'
            # print(path)

            slice_ind = self.data.loc[index, 'slice']
            tonic = self.data.loc[index, 'tonic']
            y_tonic = self.freq_to_cents(tonic, 25)
            y_tonic = np.reshape(y_tonic, [-1, 6, 60])
            y_tonic = np.sum(y_tonic, axis=1)
            # y_tonic = np.array([y_tonic])
            pitches, cqt = self.__data_generation_raga(index, path, mbid, slice_ind)
            # X, pitches = self.__data_generation_pitch(index, path, slice_ind)
            # X = [X]
            pitches = np.array([pitches])
            # X = self.__data_generation(path, slice_ind)
            label = self.data.loc[index, 'labels']
            # print('raga label: {}, index:{}'.format(label,index))
            # print('label: ', label)
            y_raga = to_categorical(label, num_classes=self.n_labels)
            y_raga = np.array([y_raga])
            # return {'x_input':X[0], 'chroma_input':X[1], 'energy_input':X[2], 'tonic_input': y_tonic} , {'tf_op_layer_tonic':y_tonic, 'raga':y_raga}
            # return {'x_input': X[0], 'chroma_input': X[1], 'energy_input': X[2], 'tonic_input': y_tonic}, {'raga': y_raga}
            # return {'pitches_input':pitches, 'tonic_input': y_tonic}, {'raga': y_raga}
            transpose_by_1 = np.random.randint(20, dtype=np.int32)
            transpose_by_2 = np.random.randint(20, dtype=np.int32)
            # bce = self.get_bce_tonic(y_tonic, transpose_by_1, transpose_by_2)
            transpose_by_1 = np.expand_dims(transpose_by_1,0)
            transpose_by_2 = np.expand_dims(transpose_by_2, 0)
            shuffle = np.array([self.shuffle], dtype=np.int32)
            # shuffle = np.array([True], dtype=np.int32)
            # shuffle = np.array([False], dtype=np.int32)
            # print(tonic)
            # shuffle = np.array([True], dtype=np.int32)
            # bce = np.expand_dims(bce,0)
            # transpose_by = 0
            # y_tonic = np.roll(y_tonic, -transpose_by, axis=1)
            # y_tonic = self.freq_to_cents(31.7)
            # transpose_by = np.array([transpose_by])
            # return {'pitches_input': pitches, 'transpose_by_1_input':np.array([transpose_by_1]), 'transpose_by_2_input':np.array([transpose_by_2])}, {'raga': y_raga, 'tf_op_layer_ExpandDims_20': np.expand_dims(bce,0), 'tf_op_layer_Roll_410':y_tonic, 'tf_op_layer_Roll_411':y_tonic}
            # return {'pitches_input': pitches, 'transpose_by_1_input': transpose_by_1, 'transpose_by_2_input': transpose_by_2}, {'tf_op_layer_ExpandDims_20': bce, 'raga': y_raga, 'raga_1': y_raga}
            # return {'pitches_input': pitches, 'transpose_by_1_input': transpose_by_1, 'transpose_by_2_input': transpose_by_2}, {'tf_op_layer_ExpandDims_20': bce, 'tf_op_layer_ExpandDims_21': np.array([0])}

            # return {'pitches_input': pitches, 'transpose_by_1_input': transpose_by_1,
            #         'transpose_by_2_input': transpose_by_2}, {'tf_op_layer_ExpandDims_20': bce}
            return {'pitches_input': pitches, 'random_input': shuffle, 'cqt_input':cqt}, {'raga': y_raga, 'tf_op_layer_tonic': y_tonic}
            # return {'pitches_input': pitches, 'random_input': shuffle, 'cqt_input': cqt}, {'raga': y_raga}
            # return {'pitches_input': pitches, 'random_input': np.array([self.shuffle], dtype=np.int32)}, {'raga': y_raga, 'tf_op_layer_tonic': y_tonic}
            # return {'pitches_input': pitches, 'tonic_input':y_tonic}, {'raga': y_raga}
            # return {'pitches_input': pitches}, {'raga': y_raga, 'tf_op_layer_ExpandDims_18': np.array([0.0]), 'tf_op_layer_ExpandDims_19': np.array([-20.0])}
            # return X, y_tonic, y_raga
        else:
            raise ValueError('Unknown task')

    def get_bce_tonic(self, y_tonic, transpose_by_1, transpose_by_2):
        y_tonic_1 = np.roll(y_tonic, -transpose_by_1, axis=1)
        y_tonic_2 = np.roll(y_tonic, -transpose_by_2, axis=1)
        return self.cross_entropy(y_tonic_1, y_tonic_2)

    # calculate cross entropy
    def cross_entropy(self, p, q):
        return self.bce_layer([p, q]).numpy()

    # def freq_to_cents(self, freq):
    #     frequency_reference = 10
    #     c_true = 1200 * np.log2(np.array(freq) / frequency_reference)
    #     cents_mapping = np.linspace(0, 7180, 360) + 1997.3794084376191
    #
    #     target = [np.exp(-(cents_mapping - ct) ** 2 / (2 * 25 ** 2)) for ct in c_true]
    #     return np.array(target)

    def freq_to_cents(self, freq, std=25):
        frequency_reference = 10
        c_true = 1200 * math.log(freq / frequency_reference, 2)

        cents_mapping = np.linspace(0, 7180, 360) + 1997.3794084376191
        target = np.exp(-(cents_mapping - c_true) ** 2 / (2 * std ** 2))
        return target

    def on_epoch_start(self):
        pass

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation_pitch(self, index, path, slice_ind):
        pitch_path = self.data.loc[index, 'pitch_path']
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

        #audio = audio[:self.model_srate*15]
        # audio = self.mp3_to_wav(path)

        # print(audio[:100])
        audio = np.pad(audio, 512, mode='constant', constant_values=0)
        audio = audio[slice_ind * self.model_srate:(slice_ind + 1) * self.model_srate]
        # audio = audio[: self.model_srate*self.cutoff]
        hop_length = int(self.model_srate * self.step_size)
        n_frames = 1 + int((len(audio) - 1024) / hop_length)
        frames = as_strided(audio, shape=(1024, n_frames),
                            strides=(audio.itemsize, hop_length * audio.itemsize))
        frames = frames.transpose().copy()
        frames -= np.mean(frames, axis=1)[:, np.newaxis]
        frames /= (np.std(frames, axis=1)[:, np.newaxis]+1e-5)


        frames, pitches = self.get_pitch_labels(pitch_path, self.step_size, mbid, frames, slice_ind)

        # frames = frames[slice_ind * int(len(frames) / n_cutoff):(slice_ind + 1) * int(len(frames) / n_cutoff)]
        # pitches = pitches[slice_ind * int(len(frames) / n_cutoff):(slice_ind + 1) * int(len(frames) / n_cutoff)]
        # self.current_data[0] = frames
        # self.current_data[1] = pitches
        # self.current_data[2] = path




        return frames, pitches

    def __data_generation_raga(self, index, path, mbid, slice_ind):
        pitch_path = self.data.loc[index, 'pitch_path']
        # pitch_path = path.replace('audio', 'pitches')
        # pitch_path = pitch_path[:pitch_path.index('.wav')] + '.pitch'
        hop_length = int(self.model_srate * self.step_size)
        n_frames = 1 + int((self.cutoff*self.model_srate - 1024) / hop_length)
        pitches, cqt = self.get_pitch_labels(pitch_path, path, mbid, self.step_size, n_frames, slice_ind)
        return pitches, cqt
        # pitches, _, _ = self.__data_generation(path, slice_ind)
        # return pitches[0]


    def __data_generation(self, path, slice_ind):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        sr, audio = wavfile.read(path)
        if len(audio.shape) == 2:
            audio = audio.mean(1)  # make mono
        audio = self.get_non_zero(audio)

        if slice_ind is not None:
            audio_cuttoff = audio[slice_ind * sr * self.cutoff:(slice_ind + 1) * sr * self.cutoff]
        else:
            rand = np.random.randint(0, len(audio) - sr * self.cutoff)
            rand = 0
            # audio_cuttoff = audio[rand*sr * self.cutoff: (rand+1)*sr * self.cutoff]
            audio_cuttoff = audio[-rand * sr * self.cutoff:]
        if len(audio_cuttoff) < sr * self.cutoff:
            audio_cuttoff = audio[-1 * sr * self.cutoff:]
        audio = audio_cuttoff
        # audio = audio[:self.model_srate*15]
        # audio = self.mp3_to_wav(path)

        # print(audio[:100])
        audio = np.pad(audio, 512, mode='constant', constant_values=0)
        # audio = audio[: self.model_srate*self.cutoff]
        hop_length = int(self.model_srate * self.step_size)
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

        frames -= np.mean(frames, axis=1)[:, np.newaxis]
        # print(np.std(frames, axis=1)[:, np.newaxis])
        frames /= (np.std(frames, axis=1)[:, np.newaxis] + 1e-5)

        chroma = np.zeros([60, 100])
        # chroma = self.get_chroma(audio, self.model_srate, hop_length)
        frames, energy_frames = self.pad_frames(frames, self.sequence_length, energy_frames)

        # print(path)
        # assert len(frames)==int(self.sequence_length*((1 + int((16000 * self.cutoff - 1024) / hop_length))//self.sequence_length))
        frames = np.array([frames])
        chroma = np.array([chroma])
        energy_frames = np.array([energy_frames])
        # print(n_frames//self.sequence_length)
        # input()
        # frames = np.array([frames])
        # mask = np.array([mask])
        # chroma = np.array([chroma])

        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # y = np.empty((self.batch_size), dtype=int)
        #
        # # Generate data
        # for i, ID in enumerate(list_IDs_temp):
        #     # Store sample
        #     X[i,] = np.load('data/' + ID + '.npy')
        #
        #     # Store class
        #     y[i] = self.labels[ID]

        return frames, chroma, energy_frames

    # def get_pitch_labels(self, pitch_path, hop_size, n_frames, slice_ind):
    #
    #     pitch_path = pitch_path.replace('/audio/', '/features/')
    #     pitch_path = self.fix_paths(pitch_path)
    #     data = pd.read_csv(pitch_path, sep='\t')
    #     if slice_ind is None:
    #         slice_ind = np.random.randint(0, data.shape[0] - n_frames)
    #     pitches = np.zeros([n_frames, 360])
    #     jump = hop_size / 0.0044444
    #     k=0
    #     for i in range(slice_ind*n_frames, (slice_ind+1)*n_frames):
    #         a = 0
    #         b = a
    #         i1 = int(jump * i)
    #         i2 = i1 + 1
    #         if i1 < data.shape[0]:
    #             a = data.iloc[i1, 1]
    #         if i2 < data.shape[0]:
    #             b = data.iloc[i2, 1]
    #         pitches[k] = self.freq_to_cents(1e-5 + (a + b) / 2)
    #         k+=1
    #     # m = min(frames.shape[0], len(pitches))
    #     # return frames[:m], pitches[:m]
    #     return pitches

    def get_pitch_labels(self, pitch_path, audio_path, mbid, hop_size, n_frames, slice_ind):

        # pitch_path = pitch_path.replace('/audio/', '/features/')
        # pitch_path = self.fix_paths(pitch_path)
        frequency_reference = 10
        cents_mapping = np.linspace(0, 7180, 360) + 1997.3794084376191
        data = pd.read_csv(pitch_path, sep='\t')

        if self.full_length:
            slice_ind = 0
            n_frames = data.shape[0]
        if self.random:
            slice_ind = None

        values = data.values[:,0]
        pitches = np.zeros([n_frames, 360])
        k = 0
        if slice_ind is None:
            slice_ind = np.random.randint(0, data.shape[0] - n_frames+1)
            for i in range(slice_ind, slice_ind + n_frames):
                freq = 1e-4 + values[i]
                c_true = 1200 * math.log(freq / frequency_reference, 2)
                target = np.exp(-(cents_mapping - c_true) ** 2 / (2 * 25 ** 2))
                pitches[k] = target
                k += 1
            cqt = self.get_cqt(mbid, audio_path, slice_ind, False)
        else:
            if (slice_ind+1)*n_frames<data.shape[0]:
                for i in range(slice_ind*n_frames, (slice_ind+1)*n_frames):
                    if i>=data.shape[0]:
                        print('breaking')
                        break
                    freq = 1e-6 + values[i]
                    c_true = 1200 * math.log(freq / frequency_reference, 2)
                    target = np.exp(-(cents_mapping - c_true) ** 2 / (2 * 25 ** 2))
                    pitches[k] = target
                    k += 1
                cqt = self.get_cqt(mbid, audio_path, slice_ind*n_frames, False)
            else:
                for i in range(n_frames):
                    freq = 1e-6 + values[data.shape[0]+i-n_frames]
                    c_true = 1200 * math.log(freq / frequency_reference, 2)
                    target = np.exp(-(cents_mapping - c_true) ** 2 / (2 * 25 ** 2))
                    pitches[k] = target
                    k += 1
                cqt = self.get_cqt(mbid, audio_path, slice_ind * n_frames, True)

            # m = min(frames.shape[0], len(pitches))
            # return frames[:m], pitches[:m]

        return pitches, cqt

    def mp3_to_wav(self, mp3_path):

        a = AudioSegment.from_mp3(mp3_path)
        y = np.array(a.get_array_of_samples())


        if a.channels == 1:
            cutoff_ch = self.cutoff * a.frame_rate
        elif a.channels == 2:
            cutoff_ch = 2*self.cutoff * a.frame_rate
        if len(y)>= cutoff_ch:
            rand = np.random.randint(len(y)-cutoff_ch)
            y = y[rand:rand+cutoff_ch]
        else:
            c = cutoff_ch - len(y)
            y = np.pad(y, c)[c:]

        if a.channels == 2:
            y = y.reshape((-1, 2))
            y = y.mean(1)
        y = np.float32(y) / 2**15
        
        y = resample(y, a.frame_rate, 16000)
        return y
        
    def get_non_zero(self, audio):
        for i,a in enumerate(audio):
            if a!=0:
                return audio[i:]
        return audio

    def fix_paths(self, path):
        fixed_path = path.replace('&', '_')
        fixed_path = fixed_path.replace(':', '_')
        fixed_path = fixed_path.replace('\'', '_')
        return fixed_path


    def pad_frames(self, frames, sequence_length, energy_frames):
        clip_length = int(sequence_length * np.floor(len(frames) / sequence_length))
        clipped_frames = frames[:clip_length]
        energy_frames = energy_frames[:clip_length]
        return clipped_frames, energy_frames

    def get_chroma(self, audio, sr, hop_length):
        # logC = librosa.amplitude_to_db(np.abs(C))
        # plt.figure(figsize=(15, 5))
        # librosa.display.specshow(logC, sr=sr, x_axis='time', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')

        hop_length = 512
        # chromagram = librosa.feature.chroma_cqt(audio, sr=sr, hop_length=hop_length)
        # plt.figure(figsize=(15, 5))
        # librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')

        chromagram = librosa.feature.chroma_cens(audio, sr=sr, hop_length=hop_length, n_chroma=60, bins_per_octave=60)
        # plt.figure(figsize=(15, 5))
        # librosa.display.specshow(chromagram, sr=sr,x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm',bins_per_octave=60)
        # print(chromagram.shape)
        # input()
        return chromagram

    def get_cqt(self, mbid, path, slice_ind_frames, last):
        c_cqt  = self.lm_file[mbid]


        slice_ind_audio = int((slice_ind_frames - 1) * (self.step_size * self.model_srate) + 1024)
        slice_ind = int(((slice_ind_audio - 1024)/512)+1)
        if not last:
            c_cqt = c_cqt[:,slice_ind:slice_ind+self.model_srate*self.cutoff]
        else:
            c_cqt = c_cqt[:, -self.model_srate * self.cutoff:]
        # c_cqt = np.mean(c_cqt, axis=1, keepdims=True)
        return np.expand_dims(np.transpose(c_cqt),0)
        # return tf.ones([1,60], np.float32)


        sr, audio = wavfile.read(path)
        if len(audio.shape) == 2:
            audio = audio.mean(1)  # make mono
        if not last:
            audio = audio[slice_ind_audio:slice_ind_audio+self.model_srate*self.cutoff]
        else:
            audio = audio[-int(self.model_srate * self.cutoff):]
        C = np.abs(librosa.cqt(audio, sr=sr, bins_per_octave=60, n_bins=60 * 7, pad_mode='wrap',
                               fmin=librosa.note_to_hz('C1')))
        #     librosa.display.specshow(C, sr=sr,x_axis='time', y_axis='cqt', cmap='coolwarm')

        # fig, ax = plt.subplots()
        c_cqt = librosa.amplitude_to_db(C, ref=np.max)
        c_cqt = np.reshape(c_cqt, [7, 60, -1])
        c_cqt = np.mean(c_cqt, axis=0, keepdims=True)
        # c_cqt = np.mean(c_cqt, axis=1, keepdims=True)
        # img = librosa.display.specshow(c_cqt,
        #                                sr=self.model_srate, x_axis='time', y_axis='cqt_note', ax=ax, bins_per_octave=60)
        # ax.set_title('Constant-Q power spectrum')
        # fig.colorbar(img, ax=ax, format="%+2.0f dB")
        return c_cqt
