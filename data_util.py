import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import cv2
import random
from keras.utils import np_utils
from tqdm import tqdm
from sklearn.model_selection import train_test_split
class DataLoader:
    datadir="./Dataset"
    img_size=28
    classes=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    X_test=None
    X_train=None
    Y_test=None
    Y_train=None
    num_classes=None
    def __init__(self,datadir,classes):
        self.datadir=datadir
        self.classes=classes
        listd=os.listdir("./")
        if "data.npy" in listd:
            self.get_data_from_npy()
        else:
            self.create_data()
    def GetTrainingData(self):
        return self.X_train,self.Y_train
    def GetTestingData(self):
        return self.X_test,self.Y_test
    def GetNumClasses(self):
        return self.num_classes
    def create_data(self):
        training_data=[]
        for category in self.classes:
            path = os.path.join(self.datadir,category)  
            class_num = self.classes.index(category)

            for img in tqdm(os.listdir(path)):
                try:
                    img_array = cv2.imread(os.path.join(path,img))
                    img = cv2.resize(img_array, (self.img_size,self.img_size))
                    training_data.append([img, class_num])
                except Exception as e:
                    pass
        random.shuffle(training_data)
        X = []
        Y = []
        for features,label in training_data:
            X.append(features)
            Y.append(label)
        X = np.array(X).reshape(-1, self.img_size,self.img_size, 3)
        seed = 785
        (X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.20, random_state=seed)
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 3).astype('float32')
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 3).astype('float32')
        self.X_train = X_train /255
        self.X_test = X_test /255
        np.save("data.npy",np.array([self.X_train,self.Y_train,self.X_test,self.Y_test]))
        self.Y_train = np_utils.to_categorical(Y_train)
        self.Y_test = np_utils.to_categorical(Y_test)
        self.num_classes = self.Y_test.shape[1]
    def get_data_from_npy(self):
        arr=np.loads("data.npy")
        self.X_train = arr[0]
        self.X_test = arr[2]
        Y_train = arr[1]
        Y_test = arr[3]
        self.Y_train = np_utils.to_categorical(Y_train)
        self.Y_test = np_utils.to_categorical(Y_test)
        self.num_classes = self.Y_test.shape[1]
