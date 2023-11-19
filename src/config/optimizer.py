
from enum import Enum

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from typing import List, Any
from omegaconf import MISSING

class LossBalanceScheme(Enum):
    none  = 0
    light = 1
    even  = 2
    focal = 3

class OptimizerKind(Enum):
    adam     = 0
    rmsprop  = 1
    sgd      = 2
    adagrad  = 3
    adadelta = 4
    lars     = 5
    lamb     = 6

@dataclass
class LRScheduleConfig:
    name:                 str = ""
    peak_learning_rate: float = 1e-3

@dataclass
class OneCycleConfig(LRScheduleConfig):
    name:                 str = "one_cycle"
    min_learning_rate:  float = 1e-3
    decay_floor:        float = 1e-4
    decay_epochs:         int = 5

@dataclass
class WarmupFlatDecayConfig(LRScheduleConfig):
    name:                 str = "standard"
    decay_floor:        float = 1e-4
    decay_epochs:         int = 5

@dataclass
class Flat(LRScheduleConfig):
    name:                 str = "flat"

@dataclass
class Optimizer:
    lr_schedule:          LRScheduleConfig = field(default_factory = lambda : Flat())
    loss_balance_scheme: LossBalanceScheme = LossBalanceScheme.focal
    name:                    OptimizerKind = OptimizerKind.adam
    gradient_accumulation:             int = 1

cs = ConfigStore.instance()

cs.store(group="lr_schedule", name="flat",      node=Flat)
cs.store(group="lr_schedule", name="one_cycle", node=OneCycleConfig)
cs.store(group="lr_schedule", name="standard",  node=WarmupFlatDecayConfig)
cs.store(name="optimizer", node=Optimizer)
