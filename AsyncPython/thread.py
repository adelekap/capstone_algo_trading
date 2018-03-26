import threading
import time
from logger import log
import datetime
from worker import worker

if __name__ == '__main__':
    start = datetime.datetime.now()
    processes = []
    for i in range(200):
        t = threading.Thread(target=worker, args=(i,))
        processes.append(t)
        t.start()
    for p in processes:
        p.join()

    log("Took " + str(datetime.datetime.now() - start))
    log("pls")

