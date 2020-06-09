'''
Created on May 15, 2019

@author: maojj
'''
from pocket.stock import lstm_train_one
from pocket.stock import predicted_next_day
import pandas as pd

def main():
    print("enter stock main")
    
def perdict(stockid):
    #fetch data
    #fetch_one.get(stockid)
    #train
    lstm_train_one.main(stockid)
    #perdict
    predicted_next_day.main(stockid)
    
    
    
    
    return "perdict stockid %s" % stockid


def is_stock(message):
    return message.isdigit()
    
def get_stock_id(message):
    
    df = pd.read_csv("stock_basic.csv")
    symbol = int(message)
    print("---")
    #stock_id =  df.loc[df["order_book_id"].str.startswith(message), "order_book_id"].values[0]
    stock_id =  df[df["symbol"]==symbol]["ts_code"].values[0]
    print("----")
    return stock_id
    

def mystock():
    print("mystock")
    

