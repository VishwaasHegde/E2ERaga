import pyhocon
from collections import defaultdict
import os
import pandas as pd
import numpy as np
import json
from sklearn import preprocessing
from pydub import AudioSegment
from resampy import resample
from scipy.io import wavfile
from shutil import copyfile
from tqdm import tqdm
model_srate = 16000
import pydub
import utils
import argparse


def create_tonic_test_train_split(config):
    data_path = config['data_path']
    split = config['split']
    cm_paths = []
    # cm_paths.append(os.path.join(data_path, 'TonicDataset','datasets','CompMusicWorkshop2012', 'Info_file_237.tsv'))
    # cm_paths.append(os.path.join(data_path, 'TonicDataset','datasets', 'CompMusicWorkshop2012', 'Info_file_540.tsv'))
    # cm_paths.append(os.path.join(data_path, 'TonicDataset','datasets', 'ISMIR2012', 'ISMIR2012.tsv'))
    cm_paths.append(os.path.join(data_path, 'TonicDataset','datasets', 'JNMR2014','CM', 'CM1.tsv'))
    cm_paths.append(os.path.join(data_path, 'TonicDataset','datasets', 'JNMR2014', 'CM', 'CM2.tsv'))
    cm_paths.append(os.path.join(data_path, 'TonicDataset','datasets', 'JNMR2014', 'CM', 'CM3.tsv'))
    cm_paths.append(os.path.join(data_path, 'TonicDataset', 'datasets', 'JNMR2014', 'IISc', 'IISc.tsv'))
    # cm_paths.append(os.path.join(data_path, 'TonicDataset', 'datasets', 'JNMR2014', 'IITM', 'IITM1.tsv'))
    cm_paths.append(os.path.join(data_path, 'TonicDataset', 'datasets', 'JNMR2014', 'IITM', 'IITM2.tsv'))


    data_dict = defaultdict(list)
    for idx, cm_path in enumerate(cm_paths):
        data = pd.read_csv(cm_path,sep='\t')

        if idx>2:
            data['tradition'] = 'Carnatic'

        data['data_type'] = cm_path[cm_path.rindex('\\')+1:][:-4]
        data = data[['path', 'tonic(hz)', 'tradition', 'data_type']]
        data['tonic'] = data['tonic(hz)'] 
        trad_groups = data.groupby(['tradition'])

        for trad,trad_val in trad_groups:
            trad_val['path'] = trad_val['path'].map(lambda x: os.path.join(data_path,x))
            data_dict[trad].append(trad_val)

    # saraga_path = os.path.join('data', 'saraga_1.0', 'saraga1.0')
    #
    # for trad in ['Carnatic', 'Hindustani']:
    #     trad_path = os.path.join(saraga_path, trad)
    #     dirs = os.listdir(trad_path)
    #     saraga_temp_data = defaultdict(list)
    #     for d in dirs:
    #         files = os.listdir(os.path.join(trad_path, d))
    #
    #         for f in files:
    #             if f.endswith('.json'):
    #                 name = f[:f.rindex('.json')]
    #                 name = os.path.join(trad_path, d, name)
    #                 info = get_info_json(name)
    #                 if info is not None:
    #                     saraga_temp_data['path'].append(info[0])
    #                     saraga_temp_data['tonic'].append(info[1])
    #                     saraga_temp_data['tradition'].append(trad)
    #
    #     data_dict[trad].append(pd.DataFrame(saraga_temp_data))


    for k,v in data_dict.items():
        # if k=='Carnatic':
        #     continue
        v = pd.concat(v)
        v = v.drop_duplicates(subset=['path'], keep='first')
        v = v.reset_index(drop=True)
        v['tradition'] = v['tradition'].map(lambda x: x.lower())
        v['old_path'] = v['path']
        v['file_exist'] = v['path'].map(os.path.exists)
        v['len'] = v['path'].apply(lambda x: mp3_to_wav(x))
        v['path'] = v['path'].apply(lambda x: mp3_to_wav_file(x))
        # mp3_file_moved_path = os.path.join(audio_data_path, mbid+'.mp3')
        # move_files(mp3_file, mp3_file_moved_path)
        # mp3_file_path.append(mp3_file_moved_path)
        v.to_csv(k+'.tsv', sep='\t')
        # v = v.drop('tonic(hz)', axis=1)
        print(v.shape)
        # train, validate, test = train_validate_test_split(v, split[0], split[1])
        train, test = utils.split_train_test(['data_type'], v)
        train, validate = utils.split_train_test(['data_type'], train)

        train_path = os.path.join(data_path, 'TonicDataset', k, 'train.tsv')
        val_path = os.path.join(data_path,'TonicDataset', k, 'validate.tsv')
        test_path = os.path.join(data_path,'TonicDataset', k, 'test.tsv')

        train.to_csv(train_path, sep='\t', index=False)
        validate.to_csv(val_path, sep='\t', index=False)
        test.to_csv(test_path, sep='\t', index=False)

# def mp3_to_wav(mp3_path, wav_path=None):
#     print(mp3_path)
#     if wav_path is None:
#         wav_path = mp3_path[mp3_path.rindex('/') + 1:]
#         wav_path = wav_path[: wav_path.rindex('.mp3')] + '.wav'
#         wav_path = os.path.join('data', 'TonicDataset', 'audio',wav_path)
#     if os.path.exists(wav_path):
#         return wav_path
#     try:
#         sound = AudioSegment.from_mp3(mp3_path)
#         sound.export(wav_path, format="wav")
#         sr, audio = wavfile.read(wav_path)
#         if len(audio.shape) == 2:
#             audio = audio.mean(1)
#         audio = resample(audio, sr, model_srate)
#         wavfile.write(wav_path, model_srate, audio)
#         return len(audio)/model_srate
#     except pydub.exceptions.CouldntDecodeError as ex:
#         print('skipped file:', mp3_path)
#         return -1

def mp3_to_wav_file(mp3_path):
    wav_path = mp3_path[mp3_path.rindex('/') + 1:]
    wav_path = wav_path[: wav_path.rindex('.mp3')] + '.wav'
    wav_path = os.path.join('data', 'TonicDataset', 'audio', wav_path)
    return wav_path

def mp3_to_wav(mp3_path, wav_path=None):
    print(mp3_path)
    if wav_path is None:
        wav_path = mp3_path[mp3_path.rindex('/') + 1:]
        wav_path = wav_path[: wav_path.rindex('.mp3')] + '.wav'
        wav_path = os.path.join('data', 'TonicDataset', 'audio', wav_path)
    if os.path.exists(wav_path):
        try:
            sr, y = wavfile.read(wav_path)
            if len(y.shape) == 2:
                y = y.mean(1)
                wavfile.write(wav_path, model_srate, y)
            if sr!=model_srate:
                y = resample(y, sr, model_srate)
                wavfile.write(wav_path, model_srate, y)
            return len(y)/model_srate
        except:
            print('skipped file:', wav_path)
        return -1
    try:
        a = AudioSegment.from_mp3(mp3_path)
        y = np.array(a.get_array_of_samples())
        if a.channels == 2:
            y = y.reshape((-1, 2))
            y = y.mean(1)
        y = np.float32(y) / 2 ** 15
        y = resample(y, a.frame_rate, model_srate)
        wavfile.write(wav_path, model_srate, y)
        return len(y) / model_srate
    except pydub.exceptions.CouldntDecodeError as ex:
        print('skipped file:', mp3_path)
        return -1

def get_info_json(name):
    json_file = name+'.json'
    data = json.load(open(json_file))
    raga_key = 'raaga'

    if 'raags' in  data:
        raga_key = 'raags'

    if len(data[raga_key])==0:
        return None
    raag = data[raga_key][0]['name']
    ctonic_file = name+'.ctonic.txt'
    with open(ctonic_file) as f:
        tonic = f.readline().strip()
    mp3_file = name + '.mp3'
    return mp3_file, tonic, raag


def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=7):
    np.random.seed(seed)
    # df = df.reset_index()
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test

def create_raga_test_train_split(config):

    data_path = config['data_path']
    traditions = config['traditions']
    split = config['split']
    for trad in traditions:
        train_path = os.path.join(data_path, 'RagaDataset', trad, 'train.tsv')
        val_path = os.path.join(data_path, 'RagaDataset', trad, 'validate.tsv')
        test_path = os.path.join(data_path, 'RagaDataset', trad, 'test.tsv')

        # if os.path.exists(train_path) and os.path.exists(train_path) and os.path.exists(train_path):
        #     continue
        path_mbid_ragaid = os.path.join(data_path, 'RagaDataset', trad, '_info_', 'path_mbid_ragaid.txt')
        df = pd.read_csv(path_mbid_ragaid, names=['path', 'mbid', 'rag_id'], sep='\t')
        df['path'] = df['path'].map(lambda x: os.path.join(data_path, x))
        df = fetch_tonic(df, data_path)
        df = df[df['len']!=-1]
        grouped = df.groupby(['rag_id'])


        ragaId_to_ragaName_mapping = os.path.join(data_path, 'RagaDataset', trad, '_info_', 'ragaId_to_ragaName_mapping.txt')
        ragaId_to_ragaName = pd.read_csv(ragaId_to_ragaName_mapping, sep='\t', names = ['rag_id', 'rag_name'])

        ragaId_to_ragaName['labels'] = np.arange(ragaId_to_ragaName.shape[0])
        ragaId_to_ragaName = ragaId_to_ragaName.set_index(['rag_id'])

        train_list, validate_list, test_list = [], [], []
        lbl = 0
        for k,v in grouped:

            v['rag_name'] = v['rag_id'].map(lambda x: ragaId_to_ragaName.loc[x]['rag_name'])
            v['labels'] = [lbl]*v.shape[0]
            v['path'] = v['path'].map(lambda x: fix_paths(x, False))
            v = v.reset_index(drop=True)
            rag_train, rag_val, rag_test = train_validate_test_split(v, split[0], split[1])
            train_list.append(rag_train)
            validate_list.append(rag_val)
            test_list.append(rag_test)
            lbl+=1

        train_list = pd.concat(train_list)
        validate_list = pd.concat(validate_list)
        test_list = pd.concat(test_list)

        train_list.to_csv(train_path, sep='\t', index=False)
        validate_list.to_csv(val_path, sep='\t', index=False)
        test_list.to_csv(test_path, sep='\t', index=False)

def fetch_tonic(data, data_path):
    paths = data['path']
    mbids = data['mbid']
    tonic_list = []
    tonic_fine_list = []
    wav_path_list = []
    audio_data_path = os.path.join(data_path,'RagaDataset','audio')
    audio_lens = []
    # mp3_file_path = []
    for mbid, path in tqdm(zip(mbids, paths)):

        path = r''+path
        data_path = os.path.dirname(os.path.realpath(__file__))
        # feature_path = os.path.join(data_path, feature_path)
        path = fix_paths(path)
        feature_path = path.replace('/audio/', '/features/')
        tonic_path = '\\\\?\\' +os.path.join(data_path, feature_path).replace('/', '\\') + '.tonic'
        tonic_fine_path = '\\\\?\\' +os.path.join(data_path, feature_path).replace('/', '\\') + '.tonicFine'
        mp3_file = '\\\\?\\' +os.path.join(data_path, path).replace('/', '\\') + '.mp3'
        if os.path.exists(tonic_path):
            with open(tonic_path, 'r') as f:
                tonic = f.readline().strip()
                tonic_list.append(tonic)
        else:
            tonic_list.append(-1)

        if os.path.exists(tonic_fine_path):
            with open(tonic_fine_path, 'r') as f:
                tonic_fine = f.readline().strip()
                tonic_fine_list.append(tonic_fine)
        else:
            tonic_fine_list.append(tonic_list[-1])
        

        wav_file = os.path.join(audio_data_path, mbid + '.wav')
        # mp3_file_moved_path = os.path.join(audio_data_path, mbid+'.mp3')
        # move_files(mp3_file, mp3_file_moved_path)
        # mp3_file_path.append(mp3_file_moved_path)
        audio_len = mp3_to_wav(mp3_file, wav_file)
        audio_lens.append(audio_len)
        wav_path_list.append(wav_file)

    data['old_path'] = data['path']
    data['tonic'] = tonic_list
    data['tonic_fine'] = tonic_fine_list
    data['path'] = wav_path_list
    data['len'] = audio_lens
    return data

def fix_paths(path, add_mp3=False):
    fixed_path = path.replace('&', '_')
    fixed_path = fixed_path.replace(':', '_')
    fixed_path = fixed_path.replace('\'', '_')

    if add_mp3:
        fixed_path = fixed_path+'.mp3'
    return fixed_path

def fetch_by_index(rag_data, k, group_val, val_id, ragaId_to_ragaName, output_cols):
    group_val_fil = group_val[['path', 'tonic', 'tonic_fine']].reset_index()

    rag_id_list = []
    rag_name_list = []
    path_list = []
    tonic_list = []
    tonic_fine_list = []
    label_list = []

    for id in val_id:
        rag_id_list.append(k)
        rag_name_list.append(ragaId_to_ragaName[k]['rag_name'])
        path_list.append(group_val_fil.iloc[0,id])
        tonic_list.append(group_val_fil.iloc[1, id])
        tonic_fine_list.append(group_val_fil.iloc[2, id])
        label_list.append(ragaId_to_ragaName[k]['rag_id'])

    rag_data[output_cols[0]] = rag_id_list
    rag_data[output_cols[1]] = rag_name_list
    rag_data[output_cols[2]] = path_list
    rag_data[output_cols[3]] = tonic_list
    rag_data[output_cols[4]] = tonic_fine_list
    rag_data[output_cols[5]] = label_list

def shuffle_split(task, tradition):
    config = pyhocon.ConfigFactory.parse_file("experiments.conf")[task]

    processes = ['train', 'validate', 'test']
    data_paths = []
    data = []
    for p in processes:
        data_path = config[tradition+'_'+p]
        data_paths.append(data_path)
        temp = pd.read_csv(data_path, sep='\t')
        data.append(temp)
    data = pd.concat(data, axis=0)
    if task=='tonic':
        train, test = utils.split_train_test(['data_type'], data)
        train, validate = utils.split_train_test(['data_type'], train)
    else:
        train, test = utils.split_train_test(['labels'], data, 0.10)
        train, validate = utils.split_train_test(['labels'], train, 0.10)

    train.to_csv(data_paths[0], sep='\t', index=False)
    validate.to_csv(data_paths[1], sep='\t', index=False)
    test.to_csv(data_paths[2], sep='\t', index=False)


def move_files(path, mbid):
    # os.rename(path, mbid)
    copyfile(path, mbid)

def tonic_train_test_split():
    config_tonic = pyhocon.ConfigFactory.parse_file("experiments.conf")['tonic']
    create_tonic_test_train_split(config_tonic)

def raga_train_test_split():
    config_raga = pyhocon.ConfigFactory.parse_file("experiments.conf")['raga']
    create_raga_test_train_split(config_raga)

if __name__ == '__main__':
    # arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument('--task', default=False, help='prepare train test split for tonic/raga')
    #
    # p_args = arg_parser.parse_args()
    #
    # if p_args.task == 'tonic':
    #     tonic_train_test_split()
    # elif p_args.task == 'tonic':
    #     raga_train_test_split()
    # else:
    #     raise ValueError('task {} is not defined'.format(p_args.task))
    # raga_train_test_split()
    # tonic_train_test_split()
    shuffle_split('raga', 'carnatic')
