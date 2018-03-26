import multiprocessing as mp
import time, datetime
from logger import log
from worker import worker



if __name__ == '__main__':
    start = datetime.datetime.now()
    processes = []
    for i in range(200):
        p = mp.Process(target=worker,args=(i,))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

    log("Took " + str(datetime.datetime.now() - start))
    log("pls")