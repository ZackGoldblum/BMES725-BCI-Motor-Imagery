import os
import sys
import time
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, AveragePooling2D
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, BatchNormalization
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

# main directory
main_dir = os.getcwd()
# eeg data directory
eeg_dir = os.path.join(main_dir, "eeg_data")

MI_CLASSES = ["left", "right", "none"]

def to_one_sec(eeg_data, num_sec=5, samp_freq=250):
    one_sec_data = []
    for i in range(num_sec):
        eeg_data_i = eeg_data[i*samp_freq:(i+1)*samp_freq]
        one_sec_data.append(eeg_data_i)

    return one_sec_data

def create_data(data_dir):
    training_data_dict = {}

    for mi_class in MI_CLASSES:
        if mi_class not in training_data_dict:
            training_data_dict[mi_class] = []

        mi_class_dir = os.path.join(data_dir, mi_class)
        for filename in os.listdir(mi_class_dir):
            if "filtered" in filename:
                eeg_data = np.loadtxt(os.path.join(mi_class_dir, filename), delimiter=',')  # (250, 8) eeg data
                one_sec_list = to_one_sec(eeg_data, num_sec=5, samp_freq=250)
                for i in range(len(one_sec_list)):  # for each 1 second trial
                    one_sec_data = one_sec_list[i]              # (250, 8) eeg data
                    one_sec_dataT = np.transpose(one_sec_data)  # (8, 250) eeg data
                    training_data_dict[mi_class].append(one_sec_dataT)  

    print("Trials per class: " + str(int(len(os.listdir(mi_class_dir))/3)) + "\n")
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

labeled_data = create_data(data_dir=eeg_dir)

print("X shape: (num trials, num channels, num samples)")
print("y shape: (num trials, one hot labels)\n")

print("Creating labeled data...")
labeled_data_X = []
labeled_data_y = []

for X, y in labeled_data:
    labeled_data_X.append(X)
    labeled_data_y.append(y)
    
print("labeled_data_X: " + str(np.shape(labeled_data_X)))
print("labeled_data_y: " + str(np.shape(labeled_data_y)) + "\n")

print("Creating testing data...")
train_X = np.array(labeled_data_X[0:9*3*5])
train_y = np.array(labeled_data_y[0:9*3*5])
print("train_X: " + str(np.shape(train_X)))
print("train_y: " + str(np.shape(train_y)) + "\n")

print("Creating training data...")
test_X = np.array(labeled_data_X[9*3*5:12*3*5])
test_y = np.array(labeled_data_y[9*3*5:12*3*5])
print("test_X: " + str(np.shape(test_X)))
print("test_y: " + str(np.shape(test_y)) + "\n")

print("Creating validation data...")
validation_X = np.array(labeled_data_X[12*3*5:])
validation_y = np.array(labeled_data_y[12*3*5:])
print("validation_X: " + str(np.shape(validation_X)))
print("validation_y: " + str(np.shape(validation_y)) + "\n")

def square(x):
    return K.square(x)

def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000))   

def ShallowConvNet(nb_classes, Chans = 8, Samples = 250, dropoutRate = 0.5):
    # start the model
    input_main   = Input((Chans, Samples, 1))
    block1       = Conv2D(40, (1, 25), 
                                    input_shape=(Chans, Samples, 1),
                                    kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1       = Conv2D(40, (Chans, 1), use_bias=False, 
                            kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(epsilon=1e-05, momentum=0.1)(block1)
    block1       = Activation(square)(block1)
    block1       = AveragePooling2D(pool_size=(1, 75), strides=(1, 15))(block1)
    block1       = Activation(log)(block1)
    block1       = Dropout(dropoutRate)(block1)
    flatten      = Flatten()(block1)
    dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax      = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)

model = ShallowConvNet(nb_classes=3, Chans=8, Samples=250)
model.summary()

model.compile(loss="categorical_crossentropy", optimizer ="adam", metrics=["accuracy"])

epochs = 50
batch_size = 10
fitted_model = model.fit(train_X, train_y, batch_size=batch_size, epochs=50, validation_data=(validation_X, validation_y))

score = model.evaluate(test_X, test_y, batch_size=batch_size)
classification_accuracy = round(score[1]*100, 2)
print("\nClassification accuracy: " + str(classification_accuracy))

model_name = f"models/{classification_accuracy}-acc_{epochs}-epochs_{batch_size}-batchsize_{int(time.time())}.model"
model.save(model_name)
print("\nModel saved.")
print(model_name)
