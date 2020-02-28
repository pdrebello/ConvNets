#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 16:49:51 2019

@author: pratheek
"""

from keras import Sequential
from keras.layers import BatchNormalization,Conv2D,MaxPooling2D,Flatten,Dense
from keras.callbacks import ModelCheckpoint
import numpy as np
import sys

program_name = sys.argv[0]
arguments = sys.argv[1:]

#arguments = ["col341_a2_q2/train.csv","col341_a2_q2/test.csv",'outputfilename.txt']
trainfilename = arguments[0]
testfilename = arguments[1]
outputfilename = arguments[2]

X_train = np.loadtxt(trainfilename,delimiter=' ',dtype=np.float32)
Y_train = X_train[:,X_train.shape[1]-1]
X_train = X_train[:,:X_train.shape[1]-1]
X_train = X_train.reshape((X_train.shape[0],3,32,32))


X_test = np.loadtxt(testfilename,delimiter=' ',dtype=np.float32)
Y_test = X_test[:,X_test.shape[1]-1]
X_test = X_test[:,:X_test.shape[1]-1]
X_test = X_test.reshape((X_test.shape[0],3,32,32))

X_train = np.transpose(X_train,(0,2,3,1))
X_test = np.transpose(X_test,(0,2,3,1))

#Y_train = Y_train.reshape((Y_train.shape[0],1))
#X_val = Y_test[45000:].reshape((Y_test.shape[0],1))
#Y_val = Y_train[45000:]

#X_train = X_train[:45000]
#Y_train = Y_train[:45000].reshape((Y_test.shape[0],1))


model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same',input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(10, activation = 'softmax'))


model.compile(loss = 'sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
filepath = "best.hd5f"
callback = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True)
history_callback = model.fit(X_train,Y_train, batch_size = 128,epochs =30,validation_split=0.1,callbacks=[callback])
model.load_weights("best.hd5f")
Y_test = np.argmax(model.predict(X_test),axis=1)
with open(outputfilename,"w") as w:
    for i in range(Y_test.shape[0]):
        w.write(str(Y_test[i]))
        w.write("\n")

loss_history = history_callback.history["loss"]
val_loss_history = history_callback.history["val_loss"]
numpy_loss_history = np.array(loss_history)
numpy_val_loss_history = np.array(val_loss_history)
np.savetxt("loss_history.txt", numpy_loss_history, delimiter=",")
np.savetxt("val_loss_history.txt", numpy_val_loss_history, delimiter=",")