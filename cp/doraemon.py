# coding: utf-8
import aiml
import os
import readline
from pocket.stock import stock

histfile = os.path.join(os.path.expanduser("~"), ".pyhist")
try:
    readline.read_history_file(histfile)
    # default history len is -1 (infinite), which may grow unruly
    readline.set_history_length(1000)
except IOError:
    pass
import atexit
atexit.register(readline.write_history_file, histfile)



kernel = aiml.Kernel()

if os.path.isfile("bot_brain.brn"):
    kernel.bootstrap(brainFile = "bot_brain.brn")
else:
    kernel.bootstrap(learnFiles = "std-startup.xml", commands = "load aiml b")
    kernel.saveBrain("bot_brain.brn")

del os, histfile


# kernel now ready for use
while True:
    message = raw_input("send message to doraemon: ")
    if message == "quit":
        exit()
    elif message == "save":
        kernel.saveBrain("bot_brain.brn")
    else:
        bot_response = kernel.respond(message)
        #TODO 产生新的xml
        
        if stock.is_stock(message):
            stockid = stock.get_stock_id(message)
            bot_response = "%s%s" % (bot_response, stock.perdict(stockid))
        
        
        
        print bot_response