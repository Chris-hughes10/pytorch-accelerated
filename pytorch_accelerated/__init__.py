import logging
from . import _version

from accelerate.utils import set_seed
from pytorch_accelerated.trainer import Trainer, TrainerPlaceholderValues

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info("Setting random seeds")
set_seed(42)

__version__ = _version.get_versions()["version"]
