import tushare as ts
import os
import pandas as pd
 
ts.set_token('9cc8a24ea3010f79fd55824b3e0c7ac8681ee84e85fb75097e558512')
pro = ts.pro_api()


"""
data = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')

print(data)
data.to_csv("stock_basic.csv")
"""

df = pd.read_csv("stock_basic.csv")
print(df)


code_list = df['ts_code'].values
print('code_list',code_list)

for now_code in code_list:
    file_name='./data/%s.csv'%(now_code)
    if os.path.isfile(file_name):
        continue
    #df = ts.pro_bar(ts_code=now_code, adj='qfq')
    df = ts.pro_bar(ts_code=now_code, end_date='20200608')
    df.to_csv(file_name)
    print('已导出%s' % (now_code))

