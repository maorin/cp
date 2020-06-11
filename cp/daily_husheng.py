# coding: utf-8
import tushare as ts
import os
import pandas as pd
import datetime
import time
from pocket.stock import stock
 

def get_stock_basic(pro):
    if os.path.isfile("stock_basic.csv"):
        df = pd.read_csv("stock_basic.csv")
        print("已经下载了股票基础数据 stock_basic.csv 如果过期请删除此文件。")
        print(df)
    else:
        data = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
        print(data)
        data.to_csv("stock_basic.csv")
    return df

def update_data(ts, file_name, now_code,  end_date):
    print(now_code)
    df = pd.read_csv(file_name)
    head_1 = df.head(1)
    print(head_1)
    old_date_str = head_1["trade_date"].values[0]
    print(str(old_date_str))
    print(type(old_date_str))
    end = datetime.datetime.strptime(end_date, '%Y%m%d')
    old_end = datetime.datetime.strptime(str(old_date_str), '%Y%m%d')
    delta = datetime.timedelta(days=1)
    start_date = (old_end + delta)
    
    start_date_str = start_date.strftime('%Y%m%d')
    if end > start_date:
        print(start_date)
        print(end)
        
        newdf = ts.pro_bar(ts_code=now_code, start_date=start_date_str, end_date=end_date)
        print(newdf)
        time.sleep(1)
        newdf = newdf.append(df)
        newdf.to_csv(file_name)
        print('已导出%s' % (now_code))
        
    
    
    
    
    
    

def download_data(pro, ts, dd):
    d_time = datetime.datetime.strptime(str(datetime.datetime.now().date())+'17:30', '%Y-%m-%d%H:%M')
    n_time = datetime.datetime.now()
    
    if n_time > d_time:
        end_date = n_time.strftime("%Y%m%d")
    
    end_date = n_time.strftime("%Y%m%d")
    
    code_list = dd['ts_code'].values
    print('code_list',code_list)
    if os.path.isfile("follow.csv"):
        df = pd.read_csv("follow.csv", dtype={'id':str})
        data = []
        for stock_id in df["id"]:    
            now_code = stock.get_stock_id(stock_id)
            file_name='./data/%s.csv'%(now_code)
            if os.path.isfile(file_name):
                update_data(ts, file_name, now_code,  end_date)
            else:
                #df = ts.pro_bar(ts_code=now_code, adj='qfq')
                df = ts.pro_bar(ts_code=now_code, end_date=end_date)
                df.to_csv(file_name)
                print('已导出%s' % (now_code))


def download_data_all(pro, ts, dd):
    d_time = datetime.datetime.strptime(str(datetime.datetime.now().date())+'17:30', '%Y-%m-%d%H:%M')
    n_time = datetime.datetime.now()
    
    if n_time > d_time:
        end_date = n_time.strftime("%Y%m%d")
    
    end_date = n_time.strftime("%Y%m%d")
    
    code_list = dd['ts_code'].values
    print('code_list',code_list)
    
    for now_code in code_list:
        file_name='./data/%s.csv'%(now_code)
        if os.path.isfile(file_name):
            update_data(ts, file_name, now_code,  end_date)
        else:
            #df = ts.pro_bar(ts_code=now_code, adj='qfq')
            df = ts.pro_bar(ts_code=now_code, end_date=end_date)
            df.to_csv(file_name)
            print('已导出%s' % (now_code))

if __name__ == "__main__":
    ts.set_token('xxxx')
    pro = ts.pro_api()
    
    dd = get_stock_basic(pro)
    download_data(pro, ts, dd)
    
    """
    file_name='./data/000001.SZ.csv'
    update_data(ts, file_name,"000001.SZ", "20200609")
    """
    
