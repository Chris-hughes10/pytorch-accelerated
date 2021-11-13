import logging
from . import _version

from accelerate.utils import set_seed

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info("Setting random seeds")
set_seed(42)

__version__ = _version.get_versions()["version"]
