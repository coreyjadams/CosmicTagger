from enum import Enum

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from .optimizer import Optimizer

class ModeKind(Enum):
    training  = 0
    iotest    = 1
    inference = 2

@dataclass
class Mode:
    name:               ModeKind = ModeKind.training
    no_summary_images:  bool     = False

@dataclass
class Train(Mode):
    checkpoint_iteration:   int         =  500
    summary_iteration:      int         = 1
    logging_iteration:      int         = 1
    optimizer:              Optimizer   = Optimizer()

@dataclass
class Inference(Mode):
    start_index:        int  = 0
    summary_iteration:  int  = 1
    logging_iteration:  int  = 1
    weights_location:   str  = ""

cs = ConfigStore.instance()
cs.store(group="mode", name="train", node=Train)
cs.store(group="mode", name="inference", node=Inference)
