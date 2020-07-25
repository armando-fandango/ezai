import logging

def get_logger(name='ezai',
               level=logging.INFO,
               fmt='{name} -{levelname:^3.1}- {message}',
               dtfmt = '%Y-%m-%d %H:%M:%S',
               dt_stamp = False):
#    global _logger
#    if _logger is None:
    if dt_stamp:
        fmt='{asctime} - {name} -{levelname:^3.1}- {message}'
    _logger = logging.getLogger(name=name)
    if not _logger.hasHandlers():
    #logging.basicConfig(format=format)
    #logger.handlers[0].stream=sys_stdout
        formatter = logging.Formatter(fmt, dtfmt, style='{')
        handler = logging.StreamHandler()
        #logger.removeHandler(handler)
        handler.setFormatter(formatter)
        _logger.addHandler(handler)
    _logger.setLevel(level)
    return _logger

def log_and_raise(exception: Exception, logger: logging.Logger = get_logger()):
    """
    Can be used to replace "raise" when throwing an exception to ensure the logging
    of the exception. After logging it, the exception is raised.
    Parameters
    ----------
    exception
        The exception instance to be raised.
    logger
        The logger instance to log the exception type and message.
    Raises
    ------
    Exception
        The provided exception
    """

    exception_type = str(type(exception)).split("'")[1]
    message = str(exception)
    logger.error(exception_type + ": " + message)

    raise exception