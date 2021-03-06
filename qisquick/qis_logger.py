import logging


level = 0  # Will be set by the main loop based on CLI args
fmt = '%(asctime)s:%(name)s:%(levelname)s: %(message)s'
filename = 'qls.log'


def config_logger(verbosity):
    """ Provides initial setup of logging system by setting format, separating out Qiskit and other 3rd party logging
        systems, and defining the log location.

    Args:
          verbosity (int): Derived from the count of v's if passed as CLI param, or passed directly via verbosity param
            in run_experiment call.  Should be in range(4).
    """

    global level, fmt
    level = _get_logging_level(verbosity)

    # Setup basic, universal logging details
    logging.basicConfig(format=fmt,
                        level=level)

    # Filter out mpl debug messages
    mpl_logger = logging.getLogger('matplotlib.font_manager')
    mpl_logger.setLevel(logging.CRITICAL)

    # Filter out urllib messages
    url_logger = logging.getLogger('urllib3.connectionpool')
    url_logger.setLevel(logging.CRITICAL)

    # Reroute qiskit logs to their own file
    qiskit_logger = logging.getLogger('qiskit.transpiler')
    fh = logging.FileHandler('qiskit_native.log')
    qiskit_logger.addHandler(fh)

    other_qiskit_logger = logging.getLogger('qiskit.transpiler.runningpassmanager')
    other_qiskit_logger.addHandler(fh)


def get_module_logger(name, file=filename, lvl=level):
    """" Should be called by each module to ensure the logger can correctly trace message sources"""

    logger = logging.getLogger(name)
    logger.setLevel(lvl)
    main_fh = logging.FileHandler(filename=file)
    main_formatter = logging.Formatter(fmt)
    main_fh.setFormatter(main_formatter)
    logger.addHandler(main_fh)
    return logger


def _get_logging_level(verbosity):
    logging_levels = {0: logging.CRITICAL,
                      1: logging.ERROR,
                      2: logging.WARNING,
                      3: logging.INFO,
                      4: logging.DEBUG}

    v_level = min(4, verbosity + 1) if verbosity is not None else 1
    return logging_levels[v_level]