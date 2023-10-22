import logging
import os


def get_logger(out_path):
    # logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger('log')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    file_handler = logging.FileHandler(os.path.join(out_path, 'log_record.log'))
    file_handler.setLevel(level = logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger