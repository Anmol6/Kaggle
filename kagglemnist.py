from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
labels = train.ix[:,0].values.astype('int32')
trainX = (train.ix[:,1:].values).astype('float32')
testX = (pd.read_csv('test.csv').values).astype('float32')

#convert labels to class matrix (needed for cross entropy loss function)
trainY = np_utils.to_categorical(labels)

#Normalize all values
trainX = (trainX - np.mean(trainX))/np.std(trainX)
testX = (testX - np.mean(testX))/np.std(testX)

input_dim = trainX.shape[1]
num_classes = trainY.shape[1]
#build the model
model = Sequential()
model.add(Dense(128,input_dim =input_dim))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

#use cross entropy loss function and rmsprop for optimization

model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop')

#Train model
model.fit(trainX,trainY, np_epoch = 10, batch_size = 16, validation_split = 0.15, show_accuracy = True, verbose=2)

preds = model.predict_classes(testX,verbose = 0)

np.savetxt('submission_cnn.csv', np.c_[range(1,len(preds)+1),preds], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
          

