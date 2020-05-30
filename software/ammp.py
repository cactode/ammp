import numpy as np

from objects import EndMill, Conditions, MachineChar
from cut import Cut
from ml import LinearModel
from optimize import Optimizer

import logging
logging.basicConfig(level="INFO")