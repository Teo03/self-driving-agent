import os

import pandas as pd

import tensorflow as tf
import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import TensorBoard

from sklearn.model_selection import train_test_split

from image_preprocess import Preprocess


class Model:

    def __init__(self, dataPath, bSize, perEpoch, valSteps, epochs):
        self.imageFilesPath = dataPath + '\\image_data\\'
        self.data = pd.read_csv(dataPath + 'data.csv')

        self.X = self.data['imageName']
        self.y = self.data['steeringAngle']

        self.batch_size = bSize
        self.stepsPerEpoch = perEpoch
        self.valSteps = valSteps
        self.epochs = epochs

    def splitData(self):
        X_train, X_valid, y_train, y_valid = train_test_split(self.X,
                                                              self.y,
                                                              test_size=0.2,
                                                              shuffle=True)

        return X_train, X_valid, y_train, y_valid

    @staticmethod
    def __initModel():
        # based on the nvidia "End to End Learning for Self-Driving Cars" paper
        model = Sequential(name='model')

        model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu', input_shape=(66, 200, 3)))
        model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu')) 
        model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu')) 
        model.add(Conv2D(64, (3, 3), activation='elu')) 
        model.add(Dropout(0.2))
        model.add(Conv2D(64, (3, 3), activation='elu')) 
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(100, activation='elu'))
        model.add(Dense(50, activation='elu'))
        model.add(Dense(10, activation='elu'))
        model.add(Dense(1)) 

        model.compile(loss='mse', optimizer=Adam(lr=1e-3))

        return model

    def train(self, modelsPath):
        # start the training of the model

        # initialize variables
        X_train, X_valid, y_train, y_valid = self.splitData()
        model = self.__initModel()
        tensorboard = TensorBoard(log_dir='logs')
        preprocess = Preprocess(self.imageFilesPath)

        # create a checkpoint to return the current bes version
        checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=os.path.join(modelsPath, 'model_checkpoint.h5'),
                                                              verbose=1,
                                                              save_best_only=True)

        # start training
        print('training started...')
        model.fit(preprocess.image_data_generator(X_train, y_train, 
                  batch_size=self.batch_size,
                  is_training=True),
                  validation_data=preprocess.image_data_generator(X_valid, y_valid,
                  batch_size=self.batch_size,
                  is_training=False),
                  steps_per_epoch=self.stepsPerEpoch,
                  epochs=self.epochs,
                  validation_steps=self.valSteps,
                  verbose=1,
                  shuffle=1,
                  callbacks=[checkpoint_callback, tensorboard])

        # save the model after the training finishes
        model.save(os.path.join(modelsPath, 'model_final.h5'))
        print('model is saved!')