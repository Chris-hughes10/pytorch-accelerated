import logging

from accelerate.state import AcceleratorState

from . import _version

from accelerate import notebook_launcher as accelerate_notebook_launcher
from accelerate.utils import set_seed
from pytorch_accelerated.trainer import Trainer, TrainerPlaceholderValues

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

# initialise singleton
state = AcceleratorState(_from_accelerator=True)

logger.info("Setting random seeds")
set_seed(42, device_specific=True)
notebook_launcher = accelerate_notebook_launcher

__version__ = _version.get_versions()["version"]
