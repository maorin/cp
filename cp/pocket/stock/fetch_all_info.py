# coding: utf-8

from rqalpha.api import *
from rqalpha import run_func
import numpy as np
import pandas as pd
import datetime
import os
import shutil
"""
Bar(symbol: u'\u73e0\u6c5f\u94a2\u7434', order_book_id: u'002678.XSHE', datetime: datetime.datetime(2014, 1, 2, 0, 0), 
open: 7.08, close: 7.07, high: 7.14, low: 7.03, volume: 3352317.0, total_turnover: 23756852, limit_up: 7.78, limit_down: 6.36)

total_turnover  资产周转率
volume 成交量

"""

pd.set_option('display.width', 100000)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


    
# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def init_dl(context):
    # 在context中保存全局变量
    context.s1 = context.config.stock_id
    context.config
    context.all_close_price = {}
    context.today = None
    
    logger.info("RunInfo: {}".format(context.run_info))
    df = (all_instruments('CS'))
    print df
    df.to_csv ("df.csv",encoding="utf-8")
    context.all = df["abbrev_symbol","order_book_id","symbol"]
    
# before_trading此函数会在每天策略交易开始前被调用，当天只会被调用一次
def before_trading_dl(context):
    logger.info("开盘前执行before_trading函数")

# 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_bar_dl(context, bar_dict):
    logger.info("每一个Bar执行")
    logger.info("打印Bar数据：")

# after_trading函数会在每天交易结束后被调用，当天只会被调用一次
def after_trading_dl(context):
    logger.info("收盘后执行after_trading函数")


def end_dl(context):
    logger.info("用户程序执行完成")


before_yesterday = (datetime.date.today() -  datetime.timedelta(days=6)).strftime("%Y-%m-%d")
yesterday = (datetime.date.today() -  datetime.timedelta(days=3)).strftime("%Y-%m-%d")
today = datetime.date.today().strftime("%Y-%m-%d")
print before_yesterday
print yesterday


def run(id):
    config_dl = {
      "stock_id":"%s.XSHG" % id,
      "base": {
        "start_date": yesterday,
        "end_date": today,
        "accounts": {
            "stock": 100000
        }
      },
      "extra": {
        "log_level": "verbose",
      },
      "mod": {
        "sys_analyser": {
          "enabled": True,
          "plot": False
        }
      }
    }
    
    # 您可以指定您要传递的参数
    run_func(init=init_dl, before_trading=before_trading_dl, handle_bar=handle_bar_dl, end=end_dl, config=config_dl)
    
if __name__=='__main__':
    run("600446")
