
from enum import Enum

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from typing import List, Any
from omegaconf import MISSING

class LossBalanceScheme(Enum):
    none  = 0
    light = 1
    even  = 2
    focal = 3

class OptimizerKind(Enum):
    adam    = 0
    rmsprop = 1

# class LRUnit(Enum):
#     iteration = 0
#     epoch     = 1

# class LRFunction(Enum):
#     linear    = 0
#     flat      = 1
#     decay     = 2

# @dataclass
# class LRSegment():
#     length: int  = 1
#     start: float = 0.0
#     end:   float = 0.0
#     function: LRFunction = LRFunction.linear

# @dataclass
# class LRSchedule():
#     units: LRUnit = LRUnit.iteration
#     schedule: List(LRSegment) = list(LRSegment())


@dataclass
class Optimizer:
    learning_rate:         float             =  0.0003
    loss_balance_scheme:   LossBalanceScheme = LossBalanceScheme.light
    name:                  OptimizerKind     = OptimizerKind.adam
    gradient_accumulation: int               = 1
    train_event_id:        bool              = True
    train_vertex:          bool              = False
    event_id_weight:       float             = 1.0
    vertex_weight:         float             = 1.0
    seg_weight:            float             = 0.0

cs = ConfigStore.instance()
cs.store(name="optimizer", node=Optimizer)
