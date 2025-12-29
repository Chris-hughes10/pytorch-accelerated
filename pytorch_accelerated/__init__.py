import logging

from accelerate import notebook_launcher as accelerate_notebook_launcher
from accelerate.utils import set_seed

from pytorch_accelerated.trainer import Trainer, TrainerPlaceholderValues

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info("Setting random seeds")
set_seed(42)
notebook_launcher = accelerate_notebook_launcher

# Version is managed by setuptools-scm and written to _version.py during build
try:
    from pytorch_accelerated._version import __version__
except ImportError:
    # If running from source without installation, version is unknown
    __version__ = "unknown"
