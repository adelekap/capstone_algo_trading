import time
from AsyncPython.logger import log


def worker(params):
    num, test = params
    for i in range(num):
        print(params[1])
        time.sleep(.05)
        log("I am thread " + str(params[0]) + " " + str(i))
