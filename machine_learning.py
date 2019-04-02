import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from exercise0 import clean_data

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, CuDNNLSTM, Flatten
from tensorflow.keras.callbacks import TensorBoard

# load data
Data = pd.read_excel("Exercise.xlsx", sheetname="Data")
AdStock = pd.read_excel("Exercise.xlsx", sheetname="AdStock")

# 0. Inspect the data - make sure you understand all variables and the objective of this exercise. Is the data "clean"/ meaningfull?									

names = list(Data)
print(names)
# Make functions to easily controll what is executed

Data = clean_data(Data)

variables = ["Media spend","TV","Radio","Dailies","GRP","Competitor 1 Spend","Competitor 2 Spend"]
norm_Data = tf.keras.utils.normalize(Data[variables].values, axis=1)


predicted_periods = 5
epochs = 10
batch_size = 1
name = f"predict_benefit{epochs}epochs_{predicted_periods}periods"

Y = []

for i in Data["Sales"]:
	
	for j in range(0,15):
		if 3000 + j * 500 <= i <= 3000 + (j + 1) * 500:
			Y.append(j)
			break

		else: continue
	

x_train = norm_Data[0:np.shape(norm_Data)[0] - 10]
y_train = Y[0:np.shape(norm_Data)[0] - 10]
x_test = norm_Data[np.shape(norm_Data)[0] - 10:]
y_test = Y[np.shape(norm_Data)[0] - 10:]

np.expand_dims(x_train, axis= 0)
print(np.shape(x_train))
model = Sequential()
model.add(CuDNNLSTM(16, input_shape=(x_train.shape[0:]), return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(CuDNNLSTM(16, input_shape=(x_train.shape[0:]), return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(CuDNNLSTM(16, input_shape=(x_train.shape[0:]), return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(8, activation="relu"))
model.add(Dropout(0.1))

model.add(Dense(8, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
model.compile(loss='sparse_categorical_crossentropy', 
			  optimizer=opt,
			  metrics=['accuracy'])

tensorboard = TensorBoard(log_dir=f'logs/{name}')

history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs)#, validation_data=(x_test,y_test))
