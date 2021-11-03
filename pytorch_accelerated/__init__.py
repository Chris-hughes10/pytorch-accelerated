import logging

from accelerate.utils import set_seed

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info('Setting random seeds')
set_seed(42)
