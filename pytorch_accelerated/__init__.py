import logging

from accelerate import notebook_launcher as accelerate_notebook_launcher
from accelerate.utils import set_seed

from pytorch_accelerated.trainer import Trainer, TrainerPlaceholderValues
from . import _version

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info("Setting random seeds")
set_seed(42)
notebook_launcher = accelerate_notebook_launcher

__version__ = _version.get_versions()["version"]
