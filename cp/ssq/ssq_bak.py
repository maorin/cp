# -*- coding: utf-8 -*-
import time
import numpy as np
import pandas as pd
import os
import math
import datetime
import random
import warnings
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop
from keras.models import Sequential
from eventlet import tpool
from keras import backend as K
from subprocess import call
import traceback
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from keras.models import load_model
from keras.models import model_from_json
import sys
from matplotlib import pyplot

from fetchssq import fetch_ssq

def load_data(filename, seq_len, normalise_window):
    data = []
    with open(filename, 'r') as f:
    
        lines = f.readlines()
        count = 0
        
        for line in lines:
            count = count + 1
            line_list = line.split("，")
            if len(line_list) > 8:
                l = []
                for a in line_list[1:8]:
                    l.append(int(a))
                l.append(line_list[-1].split("\n")[0])
                #print l
                data.append(l)

    data = [x for x in data if str(x) != 'nan']
    
    data.sort(key=lambda x:x[-1])
    #print data
    data = [x[:-1] for x in data]
    
    for i in data[-20:-1]:
        print i
    
    
    sequence_length = seq_len + 1
    result = []

    for index in range(len(data) - sequence_length):
        a = data[index: index+seq_len]
        b = [data[index+sequence_length]]
        row = np.concatenate((a,b))
        #print row
        result.append(row)
    
    
    
    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)

    train = result
    #np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]

    #x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    #x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    
    scaler = StandardScaler()
    #x_train_scaled = scaler.fit_transform(x_train)
    #y_train_scaled = scaler.fit_transform(y_train)
    
    #return [x_train_scaled, y_train_scaled]      
    return [x_train, y_train]      

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data
"""
def build_model(layers):
    model = Sequential()
    model.add(LSTM(50, input_shape=(layers[1], layers[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    return model
"""
def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_shape=(layers[1], layers[0]),
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.1))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.1))

    model.add(Dense(
        output_dim=layers[0]))
    model.add(Activation("linear"))

    start = time.time()
    #model.compile(loss="mse", optimizer="rmsprop")
    model.compile(loss='mse', optimizer='adam')
    #model.compile(loss="mse", optimizer=RMSprop(lr=0.003, rho=0.9, epsilon=1e-06))
    print("> Compilation Time : ", time.time() - start)
    return model
 
def main():
    
    global_start_time = time.time()
    epochs  = 1
    seq_len = 50
    
    X_train, y_train = load_data('ssq.txt' , seq_len, False)
    #return False
    print X_train
    
    print('> Data Loaded. Compiling... X_train len:%s' % len(X_train))
    if len(X_train) < 500:
        return None
    
    try:
        model = build_model([7, seq_len, 100, 1])
    except Exception as e:
        print traceback.format_exc()
    
    
    history = model.fit(
        X_train,
        y_train,
        batch_size=512,
        nb_epoch=epochs,
        validation_split=0.05,
        shuffle=False)
    
    
    #history = model.fit(X_train, y_train, epochs=50, batch_size=72, validation_data=(X_train, y_train), verbose=2, shuffle=False)

    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    print("loss: %s" % history.history['loss'])
    print("val_loss: %s"  % history.history['val_loss'])    
    """
    model.save_weights('ssq.h5')
    model_json = model.to_json()
    with open('jsonssq.h5', "w") as json_file:
        json_file.write(model_json)
    json_file.close()
    """
    
    #del model
    #K.clear_session()
    """
    model.save('ssq.h5') 
    model.reset_states()
    
    
    json_file = open("jsonssq.h5", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("ssq.sh")
    """
    
    flag = random.randint(1, 1000)
    flag = -2
    test_tran = [X_train[flag]]
    #test_tran = test_tran[newaxis,:]
    #test_tran = test_tran[:,:,newaxis]
    
    result = model.predict([test_tran])
    print result
    
    ss = StandardScaler()
    print X_train[flag][-1]
    data =  X_train[0][0][:, np.newaxis]
    std_data = ss.fit_transform(data)

    origin_data = ss.inverse_transform(result[0])
    print origin_data
    

    #print "Training duration (s) : %s  %s" % (time.time() - global_start_time)

if __name__=='__main__':
    #fetch_ssq()
    main()
