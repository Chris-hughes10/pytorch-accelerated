import logging

from accelerate.utils import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info('Setting random seeds')
set_seed(42)
