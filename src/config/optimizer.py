
from enum import Enum

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

class LossBalanceScheme(Enum):
    none  = 0
    light = 1
    even  = 2
    focal = 3

class OptimizerKind(Enum):
    adam    = 0
    rmsprop = 1




@dataclass
class Optimizer:
    learning_rate:         float             =  0.0003
    loss_balance_scheme:   LossBalanceScheme = LossBalanceScheme.focal
    name:                  OptimizerKind     = OptimizerKind.adam
    gradient_accumulation: int               = 1


cs = ConfigStore.instance()
cs.store(name="optimizer", node=Optimizer)
