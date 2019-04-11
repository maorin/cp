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
from keras.optimizers import RMSprop, Adadelta
from keras.models import Sequential
from keras import backend as K
from subprocess import call
import traceback
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from keras.models import load_model
from keras.models import model_from_json
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import sys
from matplotlib import pyplot

from fetchssq import fetch_ssq
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class SSQ():
    def __init__(self):
        self.p0 = None
        self.back_test_data = []
        self.back_test_count = 0
        
    def load_data(self, filename, seq_len, normalise_window, n):
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
                    data.append(l)
    
        data = [x for x in data if str(x) != 'nan']
        
        data.sort(key=lambda x:x[-1])
        #print data
        data = [x[:-1] for x in data]
        
        new_data = []
        for i in data:
            #print i[n]
            new_data.append(i[n])



        if self.back_test_count:
            last_data = new_data[-7 + self.back_test_count:self.back_test_count]
            data = new_data[:self.back_test_count]
        else:
            last_data = new_data[-7:]
            data = new_data        

        #data = new_data[:-1]
        #last_data = new_data[-8:-1]
        
        sequence_length = seq_len + 1
        result = []
    
        for index in range(len(data) - sequence_length):
            a = data[index: index+seq_len]
            b = [data[index+sequence_length]]
            row = np.concatenate((a,b))
            result.append(row)
        
        if normalise_window:
            result = self.normalise_windows(result)
    
        result = np.array(result)
        #print result
        
        train = result
        np.random.shuffle(train)
        x_train = train[:, :-1]
        y_train = train[:, -1]
    
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
        return [x_train, y_train, last_data]   


    def normalise_windows(self, window_data):
        normalised_data = []
        for window in window_data:
            self.p0 = float(window[0])
            normalised_window = [((float(p) / self.p0) - 1) for p in window]
            normalised_data.append(normalised_window)
        return normalised_data
    
    def restore_normalise_windows(self, window_data):
        restore_normalise_data = []
        for window in window_data:
            restore_normalise_window = [self.p0 * (float(p) + 1) for p in window]
            restore_normalise_data.append(restore_normalise_window)
        return restore_normalise_data

    def build_model(self, layers=[1, 50, 100, 1]):
        model = Sequential()
    
        model.add(LSTM(
            input_shape=(layers[1], layers[0]),
            output_dim=layers[1],
            return_sequences=True))
        model.add(Dropout(0.2))
        """
        model.add(LSTM(
            layers[2],
            return_sequences=True))
        model.add(Dropout(0.2))
        
        model.add(LSTM(
            layers[2],
            return_sequences=True))
        model.add(Dropout(0.2))
        """
        model.add(LSTM(
            layers[2],
            return_sequences=False))
        model.add(Dropout(0.2))
    
        model.add(Dense(
            output_dim=layers[3]))
        model.add(Activation("linear"))
        #model.add(Activation("softmax"))
        #model.add(Activation("relu"))
    
        start = time.time()
        model.compile(loss="mse", optimizer="rmsprop", metrics=["accuracy"])
        #model.compile(loss="mse", optimizer=Adadelta(lr=0.003, rho=0.9, epsilon=1e-06))
        print("> Compilation Time : ", time.time() - start)
        return model

    def get_backtest_data(self, btc):
        data = []
        with open("ssq.txt", 'r') as f:
        
            lines = f.readlines()
            count = 0
            
            for line in lines:
                count = count + 1
                line_list = line.split("，")
                if len(line_list) > 8:
                    l = []
                    for a in line_list[1:8]:
                        l.append(int(a))
                    #print line_list[-2]
                    l.append(line_list[-2])
                    #print l
                    data.append(l)
    
        data = [x for x in data if str(x) != 'nan']
        
        data.sort(key=lambda x:x[-1])
        #print data
        data = [x[:-1] for x in data]
        
        new_data = []
        for i in data:
            print i
            new_data.append(i)
        data = new_data[btc:]
        return data


    def compare_data(self, r, p):
        for i in range(len(r)):
            f_ball_r = r[i][:-1]
            b_ball_r = r[i][-1:]
            
            f_ball_p = p[i][:-1]
            b_ball_p = p[i][-1:]
            
            
            f_out =  set(f_ball_r).intersection(set(f_ball_p))
            b_out =  set(b_ball_r).intersection(set(b_ball_p))
            print "红球 %s中  蓝球 %s中" % (len(f_out),len(b_out))
            print "红球%s 蓝球%s  real: %s  p: %s" % (f_out, b_out, r[i], p[i])


    def backtest(self, btc):
        self.back_test_count = btc

        r_data = self.get_backtest_data(btc)
            
        res = []
        for j in range(abs(self.back_test_count)):
            result = []
            for i in range(7):
                result.append(int(round(self.run(i))))
            print result
            res.append(result)
            self.back_test_count = self.back_test_count + 1

        
        self.compare_data(r_data, res)

    def run(self, n):
        
        global_start_time = time.time()
        epochs  = 20
        seq_len = 50
        
        X_train, y_train, last_data= self.load_data('ssq.txt' , seq_len, True, n)
        print last_data
        
        print('> Data Loaded. Compiling... X_train len:%s' % len(X_train))
        if len(X_train) < 500:
            return None
        
        try:
            model = self.build_model([1, seq_len, 100, 1])
        except Exception as e:
            print traceback.format_exc()
        
        
        history = model.fit(
            X_train,
            y_train,
            batch_size=512,
            nb_epoch=epochs,
            validation_split=0.02)
        
        
        """
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()
        """
        print("best: %s" % history.best_score_)
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
        print result[0][0]
        #return  self.restore_normalise_windows(result)[0][0]
        
        del model
        K.clear_session()
        
        print last_data
        self.normalise_windows([last_data])
        ball = self.restore_normalise_windows(result)[0][0]
        print ball
        return  ball        
        
        
        """
        ss = StandardScaler()
        print X_train[flag][-1]
        data =  X_train[0][0][:, np.newaxis]
        std_data = ss.fit_transform(data)
        origin_data = ss.inverse_transform(result[0])
        print origin_data
        """
    
        #print "Training duration (s) : %s  %s" % (time.time() - global_start_time)

    def tuning_param(self, n):
        
        global_start_time = time.time()
        epochs  = 20
        seq_len = 50
        
        X_train, y_train, last_data= self.load_data('ssq.txt' , seq_len, True, n)
        print last_data
        
        print('> Data Loaded. Compiling... X_train len:%s' % len(X_train))
        if len(X_train) < 500:
            return None
        
        
        model = KerasClassifier(build_fn=self.build_model, verbose=0)
        batch_size = [10, 20, 30, 50, 80, 100]
        epochs = [10, 20, 50, 80, 100]
        param_grid = dict(batch_size=batch_size, nb_epoch=epochs)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
        grid_result = grid.fit(X_train, y_train)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        cv = grid_result.cv_results_
        for params, mean_test_score, std_test_scores in zip(cv["params"], cv["mean_test_score"],cv["std_test_score"]):
            print("%f (%f) with: %r" % (mean_test_score, std_test_scores, params))


if __name__=='__main__':
    #fetch_ssq()
    
    SSQ().tuning_param(6)
    
    """
    result = []
    for i in range(7):
        result.append(round(SSQ().run(i)))
    print result
    """
    
        
