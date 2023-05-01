from enum import Enum

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

class DistributedMode(Enum):
    DDP     = 0
    horovod = 1

@dataclass
class Framework:
    name: str = MISSING
    seed: int = 0

@dataclass
class Tensorflow(Framework):
    name:                           str  = "tensorflow"
    inter_op_parallelism_threads:   int  = 2
    intra_op_parallelism_threads:   int  = 24

@dataclass
class Torch(Framework):
    name:             str             = "torch"
    sparse:           bool            = False
    distributed_mode: DistributedMode = DistributedMode.DDP

cs = ConfigStore.instance()
cs.store(group="framework", name="tensorflow", node=Tensorflow)
cs.store(group="framework", name="torch", node=Torch)
