import numpy as np
import time 
from utils import get_data
from utils import *
from DiscreteClassifier import DiscreteClassifier, make_data_loader

adl_data = load_disco_adls()

# Extract all data 
emg, imu, labels,myo_labels = get_data()
training_emg = np.array(emg['training'], dtype='object')
testing_emg = np.array(emg['testing'], dtype='object')
training_labels = np.array(labels['training'])
testing_labels = np.array(labels['testing'])

emg_data_all = np.hstack([training_emg, testing_emg])
labels_all = np.hstack([training_labels, testing_labels])
active_idxs = np.where(labels_all != 0)[0]
nm_idxs = np.where(labels_all == 0)[0] 

# Resize NM data 
for i in nm_idxs:
    clip_length = np.random.randint(150, 351)
    emg_data_all[i] = emg_data_all[i][0:clip_length]

# Parameters:
WINDOW_SIZE = 10 
INCREMENT_SIZE = 5
MODEL = "GRU"
LAYERS = 3 

# Updat this if you want to use handcrafted features 
emg_feats = get_features(emg_data_all, WINDOW_SIZE, INCREMENT_SIZE, None, None)
adl_feats = get_features(adl_data, WINDOW_SIZE, INCREMENT_SIZE, None, None)

# Split Dataset 
train_split = 0.95 
test_split = 0.05

train_labels = labels_all[0:int(len(labels_all)*train_split)]
test_labels = labels_all[-int(test_split*len(labels_all)):]
train_emg = emg_feats[0:int(len(emg_feats)*train_split)]
test_emg = emg_feats[-int(test_split*len(emg_feats)):]

# Add ADL data
adl_train = adl_feats[0:int(len(adl_feats) * train_split)]
adl_test  = adl_feats[-int(len(adl_feats) * test_split):]
train_labels = np.hstack([train_labels, np.zeros(len(adl_train))])
train_emg = np.hstack([train_emg, adl_train])
test_labels = np.hstack([test_labels, np.zeros(len(adl_test))])
test_emg = np.hstack([test_emg, adl_test])

# Fit Discrete  classifier
print("Fitting Discrete Classifier...")
classifier = DiscreteClassifier(train_emg[0].shape, type=MODEL, temporal_layers=LAYERS, file_name='Discrete_' + str(time.time()))
tr_dl = make_data_loader(train_emg, train_labels)
te_dl = make_data_loader(test_emg, test_labels)
classifier.fit(tr_dl, te_dl, learning_rate=1e-3, epochs=100)
