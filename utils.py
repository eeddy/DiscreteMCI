import numpy as np
import libemg
from os import walk
import random 
import pickle
import json 
import os 

gesture_mapping = {'noGesture': 0, 'fist': 1, 'waveIn': 2, 'waveOut': 3, 'open': 4, 'pinch': 5}

def extract_data(data):
    emg = np.transpose([data['emg']['ch' + str(ch)] for ch in range(1, 9)])
    quat = np.transpose([data['quaternion'][v] for v in ['w','x','y','z']])
    label = None
    myo_labels = np.array(data['myoDetection'])
    myo_labels = myo_labels[np.where(myo_labels != 0)]
    if len(myo_labels) == 0:
        myo_labels = [0]

    if 'gestureName' in data:
        label = gesture_mapping[data['gestureName']]
    else: 
        return None, None, None, None # Half of the test data isn't available 
    
    if 'groundTruth' in data:
        ground_truth = np.diff(np.array(data['groundTruth']))
        try:
            start_idx = np.where(ground_truth == 1)[0][0]
        except:
            start_idx = 0
        try:
            end_idx = np.where(ground_truth == -1)[0][0]
        except:
            end_idx = len(emg) - 1
    else: 
        start_idx = 0
        end_idx = len(emg)
    return emg[start_idx:end_idx], quat[int(start_idx * 0.25):int(end_idx*0.25)], label, mode(myo_labels)
    
def get_data(gesture_sets=['trainingSamples', 'testingSamples'], subjects=list(range(1,307)), subject_types=['training', 'testing']):
    # Check if pkl file exists (to save time)
    if os.path.exists('dataset.pkl') and len(subjects) > 300:
        with open('dataset.pkl', 'rb') as f:
            data = pickle.load(f)
            return data[0], data[1], data[2], data[3]

    emg_data = {}
    imu_data = {}
    labels = {}
    myo_labels = {}
    # Initialze dictionary:
    for t in subject_types:
        emg_data[t] = []  
        imu_data[t] = []
        labels[t] = []
        myo_labels[t] = []
    for t in subject_types:
        print("Getting " + t + " subjects...")
        for sub in subjects:
            if sub % 10 == 0:
                print("Subject " + str(sub) + "...")
            f = open('EMG-EPN612/' + t + 'JSON/user' + str(sub) + '/user' + str(sub) + '.json', encoding="utf8")
            jd = json.load(f)
            for s in gesture_sets:
                for sample in jd[s]: 
                    e,i,l,ml = extract_data(jd[s][sample])
                    if e is not None:
                        emg_data[t].append(e)
                        imu_data[t].append(i)
                        labels[t].append(l)
                        myo_labels[t].append(ml)

    # Save dataset as pkl (to save time)
    if len(subjects) > 300:
        with open('dataset.pkl', 'wb') as f:
            pickle.dump([emg_data, imu_data, labels, myo_labels], f, protocol=pickle.HIGHEST_PROTOCOL)

    return emg_data, imu_data, labels, myo_labels

def load_disco_adls():
    adl_files = []
    for s in range(1,16):
        path = 'DiscoDataset/S' + str(s) + '/ADL/'
        files = next(walk(path), (None, None, []))[2]
        for f in files:
            adl_files.append(path + f)

    adl_data = []
    for af in adl_files:
        if '.csv' in af:
            adl_data.append(np.loadtxt(af, delimiter=','))
    adl_data = np.vstack(adl_data)
    adl_data = adl_data[:,-8:]

    adl_data_w = [adl_data[i:i+random.randint(150, 400)] for i in range(0, len(adl_data)-400, 50)]
    adl_data_w = np.array(adl_data_w, dtype = 'object')
    return adl_data_w

def get_features(data, window_size, window_inc, feats, feat_dic):
    from libemg.feature_extractor import FeatureExtractor
    fe = FeatureExtractor()
    data = np.array([libemg.utils.get_windows(d, window_size, window_inc) for d in data], dtype='object')
    if feats is None:
        return data 
    if feat_dic is not None:
        feats = np.array([fe.extract_features(feats, d, array=True, feature_dic=feat_dic) for d in data], dtype='object')
    else:
        feats = np.array([fe.extract_features(feats, np.array(d, dtype='float'), array=True) for d in data], dtype='object')
    feats = np.nan_to_num(feats, copy=True, nan=0, posinf=0, neginf=0)
    # expected shape: (NFiles,) -> (Time, channel)
    return feats