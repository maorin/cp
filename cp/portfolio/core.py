import pandas as pd
import os
from pocket.stock import stock

def follow(stock_id):
    if os.path.isfile("follow.csv"):
        ff = pd.read_csv("follow.csv",  dtype={'id':str})
        df = pd.DataFrame()
        series = pd.Series({"id":str(stock_id)}, name=str(stock_id))
        df = df.append(series)
        for stock_id  in ff["id"]:
            series = pd.Series({"id":str(stock_id)},name=str(stock_id))
            df = df.append(series)
            df.to_csv("follow.csv")
    else:
        series = pd.Series({"id":str(stock_id)},name=str(stock_id))
        df = pd.DataFrame()
        df = df.append(series)
        df.to_csv("follow.csv")


def predict():
    if os.path.isfile("follow.csv"):
        df = pd.read_csv("follow.csv", dtype={'id':str})
        data = []
        for stock_id in df["id"]:
            res = stock.perdict(stock.get_stock_id(stock_id))
            data.append(res)
            
        df = pd.DataFrame(data)
        print("-dafa-fa---------------------")
        print(df)
        df.to_csv("predict.csv")
            
        
        