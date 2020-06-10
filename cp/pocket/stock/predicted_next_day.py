# coding: utf-8

import numpy as np
import pandas as pd
import datetime
import os, sys
from keras.models import load_model,model_from_json
from keras import backend as K
from numpy import newaxis
import shutil
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

print(len(sys.argv))
if len(sys.argv) == 2:
    today = sys.argv[1]
    now_time = datetime.datetime.strptime(today, "%Y%m%d")
    yesterday =  (now_time + datetime.timedelta(days=-1)).strftime("%Y-%m-%d")    
    

else:
    today = datetime.date.today().strftime("%Y-%m-%d")
    yesterday = (datetime.date.today() -  datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    

    


def handle_bar_next_day(id):
    
    result = {}
    
    
    order_book_id = id
    global_start_time = time.time()
    df = pd.read_csv("data/%s.csv"% id)
    history_close = df["close"].values
    
    history_close = history_close[:50]
    history_close = history_close[::-1]
    if len(history_close) != 50:
        print("history_close != 50 %s" % len(history_close))
        return "history_close != 5"
    if not os.path.isfile('weight_day/%s.csv.h5' % (order_book_id)):
        print("weight_day/%s.csv.h5  not file" % (order_book_id))
        return "weight_day/%s.csv.h5  not file" % (order_book_id)
    if not os.path.isfile('weight_json_day/%s.csv.h5' % (order_book_id)):
        print('weight_json_day/%s.csv.h5 not file' % (order_book_id))
        return 'weight_json_day/%s.csv.h5 not file' % (order_book_id)
    
    y = history_close
    
    yesterday_close = history_close[-1]
    normalised_history_close = [((float(p) / float(history_close[0])) - 1) for p in history_close]
    print("history_close: %s" % history_close)
    print("normalised_history_close: %s" % normalised_history_close)
    normalised_yesterday_close = normalised_history_close[-1]
    
    normalised_history_close = np.array(normalised_history_close)
    normalised_history_close = normalised_history_close[newaxis,:]
    normalised_history_close = normalised_history_close[:,:,newaxis]
    


    json_file = open("weight_json_day/%s.csv.h5"% (order_book_id), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("weight_day/%s.csv.h5" % (order_book_id))
    
    
    #model = load_model('model/%s.h5' % order_book_id)
    #model.compile(loss="mse", optimizer="rmsprop")
    print(normalised_history_close)
    predicted_result = model.predict(normalised_history_close)
    print(predicted_result)
    predicted = predicted_result[0,0]
    del model
    K.clear_session()
    normalised_history_close = [((float(p) / float(history_close[0])) - 1) for p in history_close]
    normalised_history_close.append(predicted)
    restore_normalise_window = [float(history_close[0]) * (float(p) + 1) for p in normalised_history_close]
    
    restore_predicted = restore_normalise_window[-1]
    
    #if restore_predicted > yesterday_close:
    inc = round(round(restore_predicted-yesterday_close, 2)  /  yesterday_close, 2)
    print("predicted: %s yesterday_close:%s restore_predicted:%s real: %s" %  (predicted,yesterday_close, restore_predicted, y))
    filename =  "%s.npy.h5" % order_book_id

    
    
    """"
    result[filename] = {"stock_id":order_book_id,
                        "normalised_yesterday_close":normalised_yesterday_close, 
                        "yesterday_close": yesterday_close, 
                        "predicted":predicted, 
                        "restore_predicted": restore_predicted, 
                        "inc": inc}

    print(result)
    
    df = pd.DataFrame(pd.DataFrame(result).to_dict("index"))
    #print df
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    df.to_csv ("predicted_result%s.csv" % (today_str), encoding="utf-8")
    """
    result = {"stock_id":order_book_id,
              "normalised_yesterday_close":normalised_yesterday_close, 
              "yesterday_close": yesterday_close, 
              "predicted":predicted, 
              "restore_predicted": restore_predicted, 
              "inc": inc}    
    
    
    return result
        



predicted_one = True

before_yesterday = (datetime.date.today() -  datetime.timedelta(days=2)).strftime("%Y-%m-%d")
yesterday = (datetime.date.today() -  datetime.timedelta(days=3)).strftime("%Y-%m-%d")
today = datetime.date.today().strftime("%Y-%m-%d")



print("-114141-----------")
print(yesterday)
print(today)



def main(id):
    return handle_bar_next_day(id)


if __name__ == "__main__":
    main("000001.SZ")
    
