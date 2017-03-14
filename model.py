import os
import tensorflow as tf
import pandas as pd
from keras import optimizers
from sklearn import model_selection
from generator import DataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D

def model():
    model1 = Sequential()
    model1.add(Lambda(lambda x: x/255-0.5, input_shape=(160, 320, 3)))
    model1.add(Cropping2D(((80, 20), (0, 20))))
    model1.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model1.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model1.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model1.add(Flatten())
    model1.add(Dense(100))
    model1.add(Activation('relu'))
    model1.add(Dense(50))
    model1.add(Activation('relu'))
    model1.add(Dense(10))
    model1.add(Activation('relu'))    
    model1.add(Dense(1))
    return model1

if __name__ == '__main__':
    data_dir = './udacity_data/'
    data = pd.read_csv(data_dir + 'driving_log.csv')
    data_train, data_valid = model_selection.train_test_split(data, test_size=.2)
    model = model()
    model.compile(optimizer=optimizers.Adam(lr=1e-04), loss='mean_squared_error')
    history = model.fit_generator(
        DataGenerator(data_train, data_dir)(),
        samples_per_epoch=data_train.shape[0]*2,
        nb_epoch=25,
        validation_data=DataGenerator(data_valid, data_dir, training_mode=False)(),
        nb_val_samples=data_valid.shape[0]
    )
    model.save('model.h5')
