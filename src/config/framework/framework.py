from enum import Enum

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

class DistributedMode(Enum):
    DDP     = 0
    horovod = 1

@dataclass
class Framework:
    name: str = MISSING

@dataclass
class Tensorflow(Framework):
    name: str = "tensorflow"
    checkpoint_iteration: int =  500
    inter_op_parallelism_threads: int = 2
    intra_op_parallelism_threads: int = 24
    environment_variables: {"TF_XLA_FLAGS" : "--tf_xla_auto_jit=2"} 

@dataclass
class Torch(Framework):
    name: str = "torch"
    sparse: bool = False
    distributed_mode: DistributedMode = DistributedMode.DDP

cs = ConfigStore.instance()
cs.store(group="framework", name="tensorflow", node=Tensorflow)
cs.store(group="framework", name="torch", node=Torch)
