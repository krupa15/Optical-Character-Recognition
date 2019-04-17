import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import cv2
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

class OCRnet:
    num_classes=0
    epoch=0
    batch_size=0
    model=None
    labels=[]
    def __init__(self,num_classes,labels):
        self.num_classes=num_classes
        self.labels=labels
        self.create_model()
    def create_model(self):
        self.model=Sequential([
            Conv2D(128, kernel_size=(3, 3), activation='relu',padding='same',input_shape=(28,28,3)),
            Conv2D(128, kernel_size=(3, 3), activation='relu',padding='same'),
            MaxPooling2D(pool_size=(2, 2),strides=2),
            Dropout(0.05),
            Conv2D(64, kernel_size=(1, 1), activation='relu'),
            Conv2D(64, kernel_size=(1, 1), activation='relu'),
            Conv2D(64, kernel_size=(1, 1), activation='relu'),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2),strides=1),
            Dropout(0.05),
            Conv2D(32, kernel_size=(3, 3), activation='relu'),
            Conv2D(32, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2),strides=2),
            Dropout(0.05),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])
    def train_model(self,epoch,batch_size,X_train,Y_train,X_test,Y_test):
        assert(self.model)
        self.epoch=epoch
        self.batch_size=batch_size                
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(X_train,Y_train, validation_data=(X_test, Y_test), epochs=self.epoch, batch_size=self.batch_size, verbose=2)
    
    def test_model(self,X_test,Y_test):
        val_loss, val_acc = self.model.evaluate(X_test, Y_test)
        return val_loss, val_acc
    
    def save_model(self,pathToStoreModel):
        assert(self.model)
        self.model.save(pathToStoreModel)

    def load_model(self,pathToLoadModel):
        self.model=tf.keras.models.load_model(pathToLoadModel)
    
    def predict_image(self,image):
        image=cv2.resize(image,(28,28))
        image=np.array([image])
        predictions = self.model.predict(image)
        tf.keras.backend.clear_session()
        print(predictions)
        return predictions[0][np.argmax(predictions)].item(),self.labels[np.argmax(predictions)]
