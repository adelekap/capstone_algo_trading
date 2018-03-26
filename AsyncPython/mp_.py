import multiprocessing as mp
import time, datetime
from logger import log
from worker import worker


def worker(stock, params):
    log("starting " + stock)
    time.sleep(10)
    log("done with " + stock + " " + str(params))


if __name__ == '__main__':
    start = datetime.datetime.now()
    processes = []
    workerArgs = [["aapl", 1], ["msft", 2], ["amd", 3], ["brk.b", 4]]
    for i in workerArgs:
        p = mp.Process(target=worker, args=(*i,))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

    log("Took " + str(datetime.datetime.now() - start))
