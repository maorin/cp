'''
Created on May 15, 2019

@author: maojj
'''
import fetch_one
import lstm_train_one

def main():
    print "enter stock main"
    
def perdict(stockid):
    
    #fetch data
    fetch_one.get(stockid)
    #train
    lstm_train_one.main(stockid)
    #perdict
    
    
    
    
    
    return "perdict stockid %s" % stockid


def is_stock(message):
    return True
    


if __name__ == '__main__':
    main()