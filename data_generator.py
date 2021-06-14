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
from tensorflow.keras.layers import Lambda
import tensorflow as tf
import h5py
import raga_feature

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
        if task=='raga':
            self.n_labels = config[tradition+'_n_labels']
        self.cutoff = config['cutoff']
        self.tradition = tradition
        self.sequence_length = int((config['sequence_length'] * self.model_srate - 1024) / hop_length) + 1
        data_path = config[tradition+'_'+process]

        data = pd.read_csv(data_path, sep='\t')
        # data = data.iloc[15:16,:]

        if full_length:
            if task=='raga':
                pitch_col = 'path'
                # pitch_col = 'old_path'
            else:
                pitch_col = 'old_path'

            if process=='train' or process=='test' or process=='validate':
                if task=='raga':
                    if process=='validate':
                        data = pd.concat([data], axis=0)
                        data = data.reset_index(drop=True)
                    elif process=='train':
                        data = pd.concat([data, data, data, data, data], axis=0)
                        data = data.reset_index(drop=True)
                    else:
                        # data = pd.concat([data, data, data], axis=0)
                        data = pd.concat([data], axis=0)
                        data = data.reset_index(drop=True)
                else:
                    # data = pd.concat([data, data, data], axis=0)
                    data = data.reset_index(drop=True)
            data['pitch_path'] = data[pitch_col].apply(lambda x: self.pitch_path(x, old=True))
            data['id']= np.arange(data.shape[0])
            data['slice'] = 0

            if task=='tonic':
                data['mbid'] = data['path']
        else:
            if task=='raga':
                data = self.get_split_data_raga(data, self.cutoff)
            else:
                data = self.get_split_data_tonic(data, self.cutoff)

        # data = data.set_index('mbid')
        self.list_IDs = data['path']
        self.data = data
        self.shuffle = shuffle
        self.random = random
        self.full_length = full_length
        self.all_notes, self.c_note = self.get_all_smooth_notes()
        bce = tf.keras.losses.BinaryCrossentropy()
        self.bce_layer = Lambda(lambda tensors: bce(tensors[0], tensors[1]))

        self.lm_file = h5py.File(config[tradition+'_cqt_cache'], "r")
        self.process = process
        self.raga_feat = np.zeros([12, 12, 60, 4])
        # self.indexes = np.arange(len(self.list_IDs))
        self.on_epoch_end()

    def get_all_smooth_notes(self):
        c_note = self.freq_to_cents_1(31.7 * 2, reduce=True)
        all_notes = np.zeros([60, 60])
        for p in range(60):
            all_notes[p] = self.get_smooth_note(c_note, p)

        return all_notes, c_note

    def freq_to_cents_1(self, freq, std=25, reduce=False):
        frequency_reference = 10
        c_true = 1200 * math.log(freq / frequency_reference, 2)

        cents_mapping = np.linspace(0, 7180, 360) + 1997.3794084376191
        target = np.exp(-(cents_mapping - c_true) ** 2 / (2 * std ** 2))

        if reduce:
            return np.sum(target.reshape([6, 60]), 0)
        return target

    def get_smooth_note(self, c_note, note):
        return np.roll(c_note, note, axis=-1)

    def pitch_path(self, path, old=False):
        #     if old_path:
        #         print(old_path)
        #         pp = old_path.replace('audio', 'features')
        #         pp = pp+'.pitch'
        # #         pp = pp[:pp.index('.wav')] + '.pitch'
        #
        #         return pp

        if '.wav' in path:
            if old:
                pp = path.replace('audio', 'pitches_orig')
                pp = pp[:pp.index('.wav')] + '.tsv'
            else:
                pp = path.replace('audio', 'pitches')
                pp = pp[:pp.index('.wav')] + '.pitch'
            # pp = path.replace('audio', 'features')

            # pp = pp[:pp.index('.wav')] + '.tsv'
        else:
            pp = path.replace('audio', 'features')
            pp = pp[:pp.index('.mp3')] + '.pit.txt'
        return pp


    def get_split_data_raga(self, data, cutoff):

        all_data = []
        id = 0
        for path, old_path, file_len, tonic, mbid, label in data[['path', 'old_path', 'len', 'tonic_fine', 'mbid', 'labels']].values:
            n_cutoff = int(file_len/cutoff)
            # n_cutoff = 1000
            # cutoffs = np.random.randint(0,n_cutoff-1,1000)
            for i in range(n_cutoff):
                json_data = {}
                json_data['id'] = id
                json_data['path'] = path
                pitch_path = self.pitch_path(path, old=True)
                json_data['pitch_path'] = pitch_path
                json_data['slice'] = i
                json_data['tonic'] = tonic
                json_data['mbid'] = mbid
                json_data['labels'] = label
                all_data.append(json_data)
                id+=1
        df = pd.DataFrame(all_data)
        df = df.set_index('id')
        return df

    def get_split_data_tonic(self, data, cutoff):

        all_data = []
        id = 0
        for path, tonic, old_path, file_len in data[['path', 'tonic', 'old_path', 'len']].values:
            n_cutoff = int(file_len/cutoff)
            # n_cutoff = 1000
            # cutoffs = np.random.randint(0,n_cutoff-1,1000)
            for i in range(n_cutoff):
                json_data = {}
                json_data['id'] = id
                json_data['path'] = path
                pitch_path = self.pitch_path(old_path)
                json_data['pitch_path'] = pitch_path
                json_data['slice'] = i
                json_data['tonic'] = tonic
                json_data['mbid'] = path
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
            mbid = self.data.loc[index, 'mbid']
            slice_ind = self.data.loc[index, 'slice']
            tonic = self.data.loc[index, 'tonic']
            y_tonic = self.freq_to_cents(tonic, 25)
            y_tonic = np.reshape(y_tonic, [-1, 6, 60])
            y_tonic = np.sum(y_tonic, axis=1)
            pitches, cqt = self.__data_generation_tonic(index, path, mbid, slice_ind)
            pitches = np.array([pitches])
            # shuffle = np.array([self.shuffle], dtype=np.int32)
            shuffle = np.array([False], dtype=np.int32)
            return {'pitches_input': pitches, 'random_input': shuffle, 'cqt_input': cqt}, {'tf_op_layer_tonic': y_tonic}
        elif self.task=='raga':
            # index = 15
            path = self.data.loc[index, 'path']
            mbid = self.data.loc[index, 'mbid']
            slice_ind = self.data.loc[index, 'slice']
            tonic = self.data.loc[index, 'tonic']
            y_tonic = self.freq_to_cents(tonic, 25)
            y_tonic = np.reshape(y_tonic, [-1, 6, 60])
            y_tonic = np.sum(y_tonic, axis=1)
            raga_feat, pitches, cqt = self.__data_generation_raga(index, path, mbid, y_tonic, slice_ind)
            raga_feat = np.array([raga_feat])
            pitches = np.array([pitches])
            label = self.data.loc[index, 'labels']
            y_raga = to_categorical(label, num_classes=self.n_labels)
            y_raga = np.array([y_raga])
            shuffle = np.array([self.process=='train'], dtype=np.int32)
            shuffle = np.array([False], dtype=np.int32)
            # return {'pitches_input': pitches, 'random_input': shuffle, 'cqt_input': cqt}, {'raga': y_raga, 'tf_op_layer_tonic': y_tonic}
            return {'pitches_input': pitches, 'raga_feature_input': raga_feat, 'random_input': shuffle, 'tonic_input': y_tonic}, {'raga': y_raga}
            # return {'pitches_input': pitches, 'random_input': shuffle, 'cqt_input': cqt, 'tonic_input': y_tonic}, {'raga': y_raga}
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

    def __data_generation_raga(self, index, path, mbid, y_tonic, slice_ind):
        pitch_path = self.data.loc[index, 'pitch_path']
        # pitch_path = path.replace('audio', 'pitches')
        # pitch_path = pitch_path[:pitch_path.index('.wav')] + '.pitch'
        hop_length = int(self.model_srate * self.step_size)
        n_frames = 1 + int((self.cutoff*self.model_srate - 1024) / hop_length)
        raga_feat, pitches, cqt = self.get_pitch_labels_raga(pitch_path, path, mbid, self.step_size, n_frames, y_tonic, slice_ind)
        return raga_feat, pitches, cqt

    def __data_generation_tonic(self, index, path, mbid, slice_ind):
        pitch_path = self.data.loc[index, 'pitch_path']
        # pitch_path = path.replace('audio', 'pitches')
        # pitch_path = pitch_path[:pitch_path.index('.wav')] + '.pitch'
        hop_length = int(self.model_srate * self.step_size)
        n_frames = 1 + int((self.cutoff*self.model_srate - 1024) / hop_length)
        pitches, cqt = self.get_pitch_labels_tonic(pitch_path, path, mbid, self.step_size, n_frames, slice_ind)
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

    def get_pitch_labels_tonic(self, pitch_path, audio_path, mbid, hop_size, n_frames, slice_ind):
        # audio_path = 'data\\TonicDataset\\audio\\15-mangalam.wav'
        # pitch_path = 'data\\TonicDataset/IITM/audio/tonic_data2/tmkkamboji2folder/1-introduction.mp3'
        st = 10.766601562499999778132305145264
        # pitch_path = pitch_path.replace('/audio/', '/features/')
        # pitch_path = self.fix_paths(pitch_path)
        frequency_reference = 10
        cents_mapping = np.linspace(0, 7180, 360) + 1997.3794084376191
        # data = pd.read_csv(pitch_path, sep='\t')
        data = pd.read_csv(pitch_path, sep='\t')
        if self.full_length:
            slice_ind = None
            # n_frames = data.shape[0]
        if self.random:
            slice_ind = None

        if data.shape[1]==2:
            values = []
            k=0
            while k < data.shape[0]:
                values.append(data.iloc[int(k),1])
                k=k+st
            data = pd.DataFrame(values)
        else:
            values = data.values[:,0]
        # values = values[values!=0]
        pitches = np.zeros([n_frames, 60])

        k = 0
        if slice_ind is None:
            if n_frames>=data.shape[0]:
                n_frames = data.shape[0]
                values = data.values[:, 0]
                pitches = np.zeros([n_frames, 60])
            slice_ind = np.random.randint(0, data.shape[0] - n_frames+1)
            for i in range(slice_ind, slice_ind + n_frames):
                if values[i]!=0:
                    freq = 1e-4 + values[i]
                    c_true = 1200 * math.log(freq / frequency_reference, 2)
                    target = np.exp(-(cents_mapping - c_true) ** 2 / (2 * 25 ** 2))
                    target = np.sum(np.reshape(target, [6,60]), axis=0)
                    pitches[k] = target
                    # pitches[k] = np.tile(data.iloc[i:i+1,:]/6, (1,6))[0]
                    # pitches[k] = data.iloc[i,:]
                k += 1
            cqt = self.get_cqt(mbid, audio_path, slice_ind, False)
        else:
            if (slice_ind+1)*n_frames<data.shape[0]:
                for i in range(slice_ind*n_frames, (slice_ind+1)*n_frames):
                    if i>=data.shape[0]:
                        print('breaking')
                        break
                    if values[i] != 0:
                        freq = 1e-6 + values[i]
                        c_true = 1200 * math.log(freq / frequency_reference, 2)
                        target = np.exp(-(cents_mapping - c_true) ** 2 / (2 * 25 ** 2))
                        target = np.sum(np.reshape(target, [6,60]), axis=0)
                        pitches[k] = target
                        # pitches[k] = data.iloc[i, :]
                    k += 1
                cqt = self.get_cqt(mbid, audio_path, slice_ind*n_frames, False)
            else:
                for i in range(n_frames):
                    if values[i] != 0:
                        freq = 1e-6 + values[data.shape[0]+i-n_frames]
                        c_true = 1200 * math.log(freq / frequency_reference, 2)
                        target = np.exp(-(cents_mapping - c_true) ** 2 / (2 * 25 ** 2))
                        target = np.sum(np.reshape(target, [6,60]), axis=0)
                        pitches[k] = target
                        # pitches[k] = data.iloc[i, :]
                    k += 1
                cqt = self.get_cqt(mbid, audio_path, slice_ind * n_frames, True)

            # m = min(frames.shape[0], len(pitches))
            # return frames[:m], pitches[:m]
        pitches = pitches[np.sum(pitches, axis=1)!=0]

        if len(pitches)==0 or len(cqt[0])==0:
            print(pitch_path)
        return pitches, cqt

    def get_pitch_labels_raga(self, pitch_path, audio_path, mbid, hop_size, n_frames, y_tonic, slice_ind):

        # pitch_path = pitch_path.replace('/audio/', '/features/')
        # pitch_path = self.fix_paths(pitch_path)
        # import time
        # start = time.time()
        frequency_reference = 10
        cents_mapping = np.linspace(0, 7180, 360) + 1997.3794084376191
        # data = pd.read_csv(pitch_path, sep='\t')
        data = pd.read_csv(pitch_path, sep=',')
        if self.full_length:
            slice_ind = None
            # n_frames = data.shape[0]
        if self.random:
            slice_ind = None

        values = data.values[:,0]
        pitches = np.zeros([1, 60])

        # pitches = np.zeros([data.shape[0], 60])

        k = 0
        if slice_ind is None:
            if n_frames>=data.shape[0]:
                n_frames = data.shape[0]
                # values = data.values[:, 0]
                # pitches = np.zeros([n_frames, 60])
            slice_ind = np.random.randint(0, data.shape[0] - n_frames+1)
        #     for i in range(slice_ind, slice_ind + n_frames):
        #         freq = 1e-4 + values[i]
        #         c_true = 1200 * math.log(freq / frequency_reference, 2)
        #         target = np.exp(-(cents_mapping - c_true) ** 2 / (2 * 25 ** 2))
        #         target = np.sum(np.reshape(target, [6,60]), axis=0)
        #         pitches[k] = target
        #         # pitches[k] = np.tile(data.iloc[i:i+1,:]/6, (1,6))[0]
        #         # pitches[k] = data.iloc[i,:]
        #         k += 1
        #     cqt = self.get_cqt(mbid, audio_path, slice_ind, False)
        # else:
        #     if (slice_ind+1)*n_frames<data.shape[0]:
        #         for i in range(slice_ind*n_frames, (slice_ind+1)*n_frames):
        #             if i>=data.shape[0]:
        #                 print('breaking')
        #                 break
        #             freq = 1e-6 + values[i]
        #             c_true = 1200 * math.log(freq / frequency_reference, 2)
        #             target = np.exp(-(cents_mapping - c_true) ** 2 / (2 * 25 ** 2))
        #             target = np.sum(np.reshape(target, [6,60]), axis=0)
        #             pitches[k] = target
        #             # pitches[k] = data.iloc[i, :]
        #             k += 1
        #         cqt = self.get_cqt(mbid, audio_path, slice_ind*n_frames, False)
        #     else:
        #         for i in range(n_frames):
        #             freq = 1e-6 + values[data.shape[0]+i-n_frames]
        #             c_true = 1200 * math.log(freq / frequency_reference, 2)
        #             target = np.exp(-(cents_mapping - c_true) ** 2 / (2 * 25 ** 2))
        #             target = np.sum(np.reshape(target, [6,60]), axis=0)
        #             pitches[k] = target
        #             # pitches[k] = data.iloc[i, :]
        #             k += 1
        #         cqt = self.get_cqt(mbid, audio_path, slice_ind * n_frames, True)

            # m = min(frames.shape[0], len(pitches))
            # return frames[:m], pitches[:m]

        # pitches = np.roll(pitches, -np.argmax(y_tonic[0]), 1)
        # pitches_arg = np.argmax(pitches, axis=1)
        # raga_feat = raga_feature.get_all_raga_features(pitches_arg, np.ones_like(pitches_arg), self.c_note,  self.all_notes)
        # end = time.time()

        raga_feat = raga_feature.get_raga_feat_cache(mbid, slice_ind, n_frames, self.raga_feat, self.all_notes, self.tradition)
        cqt = self.get_cqt(mbid, audio_path, slice_ind * n_frames, True)
        # raga_feat = self.gauss_smooth(raga_feat)
        # raga_feat = np.zeros([12,12,60,4])
        # raga_feat = self.stadardize(raga_feat)
        # print(end - start)
        # cqt = np.roll(cqt, -np.argmax(y_tonic[0]),2)
        # raga_feat = np.zeros([12,11,60,4])
        return raga_feat, pitches, cqt

    # def get_pitch_labels_raga(self, pitch_path, audio_path, mbid, hop_size, n_frames, slice_ind):
    #     # pitch_path = pitch_path.replace('/audio/', '/features/')
    #     pitch_path = self.fix_paths(pitch_path)
    #     st = 7.0625
    #     # pitch_path = pitch_path.replace('/audio/', '/features/')
    #     # pitch_path = self.fix_paths(pitch_path)
    #     frequency_reference = 10
    #     cents_mapping = np.linspace(0, 7180, 360) + 1997.3794084376191
    #
    #     data = pd.read_csv(pitch_path, sep='\t')
    #
    #     values = []
    #     k = 0
    #     while k < data.shape[0]:
    #         values.append(data.iloc[int(k), 1])
    #         k = k + st
    #     data = pd.DataFrame(values)
    #
    #     if self.full_length:
    #         slice_ind = None
    #         # n_frames = data.shape[0]
    #     if self.random:
    #         slice_ind = None
    #
    #     values = data.values[:,0]
    #     pitches = np.zeros([n_frames, 60])
    #     k = 0
    #     if slice_ind is None:
    #         if n_frames>=data.shape[0]:
    #             n_frames = data.shape[0]
    #             values = data.values[:, 0]
    #             pitches = np.zeros([n_frames, 60])
    #         slice_ind = np.random.randint(0, data.shape[0] - n_frames+1)
    #         for i in range(slice_ind, slice_ind + n_frames):
    #             freq = 1e-4 + values[i]
    #             c_true = 1200 * math.log(freq / frequency_reference, 2)
    #             target = np.exp(-(cents_mapping - c_true) ** 2 / (2 * 25 ** 2))
    #             target = np.sum(np.reshape(target, [6,60]), axis=0)
    #             pitches[k] = target
    #             # pitches[k] = np.tile(data.iloc[i:i+1,:]/6, (1,6))[0]
    #             # pitches[k] = data.iloc[i,:]
    #             k += 1
    #         cqt = self.get_cqt(mbid, audio_path, slice_ind, False)
    #     else:
    #         if (slice_ind+1)*n_frames<data.shape[0]:
    #             for i in range(slice_ind*n_frames, (slice_ind+1)*n_frames):
    #                 if i>=data.shape[0]:
    #                     print('breaking')
    #                     break
    #                 freq = 1e-6 + values[i]
    #                 c_true = 1200 * math.log(freq / frequency_reference, 2)
    #                 target = np.exp(-(cents_mapping - c_true) ** 2 / (2 * 25 ** 2))
    #                 target = np.sum(np.reshape(target, [6,60]), axis=0)
    #                 pitches[k] = target
    #                 # pitches[k] = data.iloc[i, :]
    #                 k += 1
    #             cqt = self.get_cqt(mbid, audio_path, slice_ind*n_frames, False)
    #         else:
    #             for i in range(n_frames):
    #                 freq = 1e-6 + values[data.shape[0]+i-n_frames]
    #                 c_true = 1200 * math.log(freq / frequency_reference, 2)
    #                 target = np.exp(-(cents_mapping - c_true) ** 2 / (2 * 25 ** 2))
    #                 target = np.sum(np.reshape(target, [6,60]), axis=0)
    #                 pitches[k] = target
    #                 # pitches[k] = data.iloc[i, :]
    #                 k += 1
    #             cqt = self.get_cqt(mbid, audio_path, slice_ind * n_frames, True)
    #
    #         # m = min(frames.shape[0], len(pitches))
    #         # return frames[:m], pitches[:m]
    #
    #     return pitches, cqt

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

    def stadardize(self, z):
        return (z - np.mean(z)) / (np.std(z)+0.001)

    def normalize(self, z):
        z_min = np.min(z)
        return (z - z_min) / (np.max(z) - z_min + 0.001)

    def gauss_smooth(self, raga_feat):
        smooth = np.zeros([12, 12, 60, 4])
        for i in range(12):
            for j in range(12):
                for k in range(4):
                    smooth[i, j, :, k] = self.gauss_smooth_util(raga_feat[i,j,:,k])
        return smooth

    def gauss_smooth_util(self, arr):
        smooth = 0
        for i in range(60):
            #         if i==57 or i==58 or i==59 or i==0 or i==1 or i==2 or i==3:
            #             continue
            #         if i==57 or i==59 or i==0 or i==1 or i==2:
            #             continue
            smooth = smooth + self.all_notes[i] * arr[i]

        smooth = self.stadardize(np.power(smooth, 1))
        return smooth

    def get_cqt(self, mbid, path, slice_ind_frames, last):
        return np.zeros([1,60, 60])
        c_cqt  = self.lm_file[mbid.lower()]
        # c_cqt = self.lm_file[mbid]

        slice_ind_audio = int((slice_ind_frames - 1) * (self.step_size * self.model_srate) + 1024)
        slice_ind = int(((slice_ind_audio - 1024)/512)+1)
        n_frames = int(((self.model_srate*self.cutoff - 1024)/512)+1)

        if not last:
            c_cqt = c_cqt[:,slice_ind:slice_ind+n_frames]
        else:
            c_cqt = c_cqt[:, -n_frames:]
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
