import core
import pyhocon
import pandas as pd
from scipy.io import wavfile
from numpy.lib.stride_tricks import as_strided
import numpy as np
import librosa
import h5py

def cache_pitch(task, tradition):
    config = pyhocon.ConfigFactory.parse_file("experiments.conf")[task]
    model = core.build_and_load_model(config, task)
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

def __data_generation_pitch(path, slice_ind, model_srate, step_size, cuttoff):
    sr, audio = wavfile.read(path)
    if len(audio.shape) == 2:
        audio = audio.mean(1)  # make mono
    audio = np.pad(audio, 512, mode='constant', constant_values=0)
    audio_len = len(audio)
    audio = audio[slice_ind * model_srate * cuttoff:(slice_ind + 1) * model_srate * cuttoff]
    if (slice_ind + 1) * model_srate * cuttoff >= audio_len:
        slice_ind = -1
    hop_length = int(model_srate * step_size)
    n_frames = 1 + int((len(audio) - 1024) / hop_length)
    frames = as_strided(audio, shape=(1024, n_frames),
                        strides=(audio.itemsize, hop_length * audio.itemsize))
    frames = frames.transpose().copy()
    frames -= np.mean(frames, axis=1)[:, np.newaxis]
    frames /= (np.std(frames, axis=1)[:, np.newaxis] + 1e-5)

    return frames, slice_ind + 1


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


def cache_cqt(task, tradition):
    config = pyhocon.ConfigFactory.parse_file("experiments.conf")[task]

    with h5py.File(config[tradition+'_cqt_cache'], "w") as f:
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
    sr, audio = wavfile.read(path)
    if len(audio.shape) == 2:
        audio = audio.mean(1)  # make mono
    C = np.abs(librosa.cqt(audio, sr=sr, bins_per_octave=60, n_bins=60 * 7, pad_mode='wrap',
                           fmin=librosa.note_to_hz('C1')))
    c_cqt = librosa.amplitude_to_db(C, ref=np.max)
    c_cqt = np.reshape(c_cqt, [7, 60, -1])
    c_cqt = np.mean(c_cqt, axis=0)
    return c_cqt
