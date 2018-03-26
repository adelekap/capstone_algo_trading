import logging

logger = logging.getLogger("main")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("log.txt")
formatter = logging.Formatter('%(threadName)s # %(threadName)s %(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)
logger.addHandler(handler)


def log(message: str):
    logger.info(message)
