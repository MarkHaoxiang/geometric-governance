from os import path
import logging

_FILE_DIRECTORY = path.dirname(path.abspath(__file__))
_BASE_DIR = path.join(_FILE_DIRECTORY, "../")

EXPERIMENT_DIR = path.join(_BASE_DIR, "experiments")
DATA_DIR = path.join(_BASE_DIR, "data")
FIGURE_DIR = path.join(_BASE_DIR, "figures")

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(name="comet_l48")
info = _logger.info
warning = _logger.warning
error = _logger.error
