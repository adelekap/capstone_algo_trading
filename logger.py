import logging

"""
The Logger writes to a log file. Can be used for all situations.
"""

class Logger(object):
    def __init__(self,filename):
        self._logger = logging.getLogger()
        self._logger.addHandler(logging.FileHandler(filename))
        self._logger.setLevel(logging.INFO)

    def log(self,message='',*args):
        self._logger.info(message,*args)
