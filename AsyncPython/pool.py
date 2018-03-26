import multiprocessing as mp
import datetime
from AsyncPython.logger import log
from AsyncPython.worker import worker

if __name__ == '__main__':
    start = datetime.datetime.now()
    p = mp.Pool(mp.cpu_count())
    p.map(worker, [[200,"test"], [300,"test2"]])
    log("Took " + str(datetime.datetime.now() - start))