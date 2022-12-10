import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Flatten, Dense
from keras.callbacks import TensorBoard
import sys
sys.path.insert(0, "/home/adam/Development/Machine_learning/Study/Sklearn/data")
from data import data, accuracy, datasets
import numpy as np

X_train, y_train, X_test, y_test = data.load_format_split(datasets.wine())
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

y_train = y_train.reshape((1279, 1))
y_test = y_test.reshape((320, 1))

#model = keras.models.load_model('data/model.h5')

model = Sequential()

model.add(Flatten(input_shape=(X_train[0].shape)))
model.add(Dense(11, activation='relu'))
model.add(Dense(22, activation='relu'))
model.add(Dense(22, activation='relu'))
model.add(Dense(11, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer="adam", loss="BinaryCrossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, batch_size=1)
model.evaluate(X_test, y_test, batch_size=1)

model.save('data/model.h5')

predictions = model.predict(X_test)
acc = accuracy.calculate(predictions, y_test)
print(acc)

