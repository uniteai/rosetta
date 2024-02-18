'''

Utility functions

'''

import logging


def mk_logger(name, level):
    ''' A logger builder helper. This helps out since Thespian is overly
    opinionated about logging.'''
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s:%(name)s => %(message)s [%(pathname)s:%(lineno)d]'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
