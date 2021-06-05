import os
import sys

import numpy as np
#sys.path.insert(1, r"C:\Users\Zack\OneDrive\BCI\arl-eegmodels-master")
#from EEGModels import ShallowConvNet

# main directory
main_dir = os.getcwd()
# eeg data directory
eeg_dir = os.path.join(main_dir, "eeg_data")
# eeg validation data directory
eeg_validation_dir = os.path.join(main_dir, "eeg_data_validation")

MI_CLASSES = ["left", "right", "none"]

print("Creating training data...")
train_data = create_data(data_dir=eeg_dir)
train_X = []
train_y = []

for X, y in train_data:
    train_X.append(X)
    train_y.append(y)

print("Creating testing data...")
test_data = create_data(data_dir=eeg_validation_dir)
test_X = []
test_y = []

for X, y in test_data:
    test_X.append(X)
    test_y.append(y)

train_X = np.array(train_X)
print("train_X shape: " + str(np.shape(train_X)))
test_X = np.array(test_X)
print("test_X shape: " + str(np.shape(test_X)))

train_y = np.array(train_y)
print("train_y shape: " + str(np.shape(train_y)))
test_y = np.array(test_y)
print("test_y shape: " + str(np.shape(test_y)))

def create_data(data_dir):
    training_data_dict = {}

    for mi_class in MI_CLASSES:
        if mi_class not in training_data_dict:
            training_data_dict[mi_class] = []

        mi_class_dir = os.path.join(data_dir, mi_class)
        for filename in os.listdir(mi_class_dir):
            if "filtered" in filename:
                eeg_data = np.loadtxt(os.path.join(mi_class_dir, filename), delimiter=',')
                one_sec_list = to_one_sec(eeg_data, num_sec=5, samp_freq=250)
                for i in range(len(one_sec_list)):  # for each 1 second trial
                    one_sec_data = one_sec_list[i]  # (250, 8) eeg data
                    one_sec_dataT = np.transpose(one_sec_data)  # (8, 250) eeg data
                    training_data_dict[mi_class].append(one_sec_dataT)  

    session_count = [len(training_data_dict[mi_class]) for mi_class in MI_CLASSES]
    print(f"One sec trial count:\nLeft: {session_count[0]}, Right: {session_count[1]}, None: {session_count[2]}\n")

    for mi_class in MI_CLASSES:
        np.random.shuffle(training_data_dict[mi_class])  # randomize session order
        training_data_dict[mi_class] = training_data_dict[mi_class][:min(session_count)]  # use min session count number of sessions

    # creating X, y 
    labeled_data = []
    for mi_class in MI_CLASSES:
        for data in training_data_dict[mi_class]:
            if mi_class == "left":
                labeled_data.append([data, [1, 0, 0]])
            elif mi_class == "right":
                labeled_data.append([data, [0, 1, 0]])
            elif mi_class == "none":
                labeled_data.append([data, [0, 0, 1]])

    np.random.shuffle(labeled_data)
    
    return labeled_data

def to_one_sec(eeg_data, num_sec=5, samp_freq=250):
    one_sec_data = []
    for i in range(num_sec):
        eeg_data_i = eeg_data[i*samp_freq:(i+1)*samp_freq]
        one_sec_data.append(eeg_data_i)

    return one_sec_data