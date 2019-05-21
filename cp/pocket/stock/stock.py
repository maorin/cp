'''
Created on May 15, 2019

@author: maojj
'''
import fetch_one
import fetch_all_info
import lstm_train_one
import predicted_next_day
import pandas as pd

def main():
    print "enter stock main"
    
def perdict(stockid):
    
    #fetch data
    fetch_one.get(stockid)
    #train
    lstm_train_one.main(stockid)
    #perdict
    predicted_next_day.main(stockid)
    
    
    
    
    return "perdict stockid %s" % stockid


def is_stock(message):
    return message.isdigit()
    
def get_stock_id(message):
    fetch_all_info.run("600446")
    
    df = pd.read_csv("df.csv")
    print "---"
    stock_id =  df.loc[df["order_book_id"].str.startswith(message), "order_book_id"].values[0]
    print "----"
    return stock_id
    

def mystock():
    print "mystock"
    


if __name__ == '__main__':
    get_stock_id("600157")