import tensorflow as tf
tf.get_logger().setLevel('WARN')

import logging
import os
from covid_xrays_model.config import config
from covid_xrays_model.config import logging_config


with open(os.path.join(config.PACKAGE_ROOT, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()


# ----------- set up logging
# MUST set this since the default for root logger is WARNING
# logging.basicConfig(level=logging.NOTSET)

logger = logging.getLogger(__name__)
# logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)
logger.addHandler(logging_config.get_console_handler())
# logger.propagate = False



