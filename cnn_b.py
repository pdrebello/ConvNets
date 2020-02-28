from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K



from keras import Sequential
from keras.layers import AveragePooling2D,Add,Activation,Input,BatchNormalization,Conv2D,MaxPooling2D,Flatten,Dense
#from keras.callbacks import ModelCheckpoint
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
import numpy as np
import sys
import keras
from keras.regularizers import l2
from keras import regularizers
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
#from classification_models.keras import Classifiers
program_name = sys.argv[0]
arguments = sys.argv[1:]
import time
start_time = time.time()





def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 0.01
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def OneHotEncode(Y):
    Ypri = np.zeros((Y.shape[0],10))
    for i in range(Y.shape[0]):
        Ypri[i][int(Y[i])] = 1
    return Ypri

print("Starting")
#arguments = ["col341_a2_q2/train.csv","col341_a2_q2/test.csv",'outputfilename.txt']
trainfilename = arguments[0]
testfilename = arguments[1]
outputfilename = arguments[2]

X_train = np.loadtxt(trainfilename,delimiter=' ',dtype=np.float32)
Y_train = X_train[:,X_train.shape[1]-1]
Y_train = OneHotEncode(Y_train)
X_train = X_train[:,:X_train.shape[1]-1]
X_train = X_train.reshape((X_train.shape[0],3,32,32))


X_test = np.loadtxt(testfilename,delimiter=' ',dtype=np.float32)
Y_test = X_test[:,X_test.shape[1]-1]

X_test = X_test[:,:X_test.shape[1]-1]
X_test = X_test.reshape((X_test.shape[0],3,32,32))

X_train = np.transpose(X_train,(0,2,3,1))
X_test = np.transpose(X_test,(0,2,3,1))

X_train = X_train/255.
X_train = X_train - 0.5
X_train = X_train * 2

X_test = X_test/255.
X_test = X_test - 0.5
X_test = X_test * 2
print("Data Loaded")


#Xtrainpri = np.flip(X_train[0:10000],axis=2)
#X_train = np.concatenate((X_train, Xtrainpri),axis=0)
#Y_train = np.concatenate((Y_train,Y_train[0:10000]),axis=0)



#Y_train = np.concatenate((Y_train,Y_train),axis=0)
#Y_train = Y_train.reshape((Y_train.shape[0],1))
#X_val = Y_test[45000:].reshape((Y_test.shape[0],1))
#Y_val = Y_train[45000:]

#X_train = X_train[:45000]
#Y_train = Y_train[:45000].reshape((Y_test.shape[0],1))




datagen = ImageDataGenerator(
    horizontal_flip=True
    )
datagen.fit(X_train)

print("Start")


model = Sequential()
weight_decay = 0

model.add(Conv2D(64, (3, 3), padding='same',
                 input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.45))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.55))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.55))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.55))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.55))

model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.55))

model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.55))

model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.55))

model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.55))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dropout(0.55))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#callback = ModelCheckpoint("best.h5", monitor='val_loss', verbose=0, save_best_only=True)
#callback2 = MyCallback()
#model.fit(X_train,Y_train, batch_size = 128,epochs =30,validation_split=0.1,callbacks=[callback])

model.fit_generator(datagen.flow(X_train, Y_train, batch_size=128),
                    steps_per_epoch = (X_train.shape[0]) / 128, epochs=40)

#model.load_weights("best.h5")

#with open("time.txt","w") as f:
#    f.write("--- %s seconds ---" % (time.time() - start_time))


Y_test = np.argmax(model.predict(X_test),axis=1)
with open(outputfilename,"w") as w:
    for i in range(Y_test.shape[0]):
        w.write(str(Y_test[i]))
        w.write("\n")