from enum import Enum

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

class DistributedMode(Enum):
    DDP       = 0
    horovod   = 1
    deepspeed = 2
    sharded   = 3

@dataclass
class Framework:
    name:    str = MISSING
    sparse: bool = False
    seed:    int = 0

@dataclass
class Tensorflow(Framework):
    name:                           str  = "tensorflow"
    inter_op_parallelism_threads:   int  = 2
    intra_op_parallelism_threads:   int  = 24

@dataclass
class Torch(Framework):
    name:             str             = "torch"
    distributed_mode: DistributedMode = DistributedMode.DDP
    oversubscribe:                int = 1

@dataclass
class Lightning(Framework):
    name:             str             = "lightning"
    distributed_mode: DistributedMode = DistributedMode.DDP
    oversubscribe:                int = 1

cs = ConfigStore.instance()
cs.store(group="framework", name="tensorflow", node=Tensorflow)
cs.store(group="framework", name="torch", node=Torch)
cs.store(group="framework", name="lightning", node=Lightning)
