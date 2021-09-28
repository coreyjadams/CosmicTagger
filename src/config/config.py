from enum import Enum

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from .network   import Network
from .mode      import Mode
from .framework import Framework
from .data      import Data

class ComputeMode(Enum):
    CPU   = 0
    GPU   = 1
    DPCPP = 2

class Precision(Enum):
    float32  = 0
    mixed    = 1
    bfloat16 = 2
    float16  = 3

# @dataclass
# class CosmicTagger:
#     network:            Network     = Network()
#     mode:               Mode        = MISSING
#     framework:          Framework   = MISSING
#     data:               Data        = MISSING

@dataclass
class Run:
    distributed:        bool        = True
    compute_mode:       ComputeMode = ComputeMode.GPU
    iterations:         int         = MISSING
    aux_iterations:     int         = MISSING
    minibatch_size:     int         = MISSING
    aux_minibatch_size: int         = MISSING
    id:                 int         = MISSING
    precision:          Precision   = Precision.float32
    profile:            bool        = False

cs = ConfigStore.instance()


cs.store(
    name="disable_hydra_logging",
    group="hydra/job_logging",
    node={"version": 1, "disable_existing_loggers": False, "root": {"handlers": []}},
)

@dataclass
class Config:
    defaults: List = field(
        default_factory=lambda: [
            {"hydra/job_logging": "disable_hydra_logging"},
        ]
    )


cs.store(name="config", node=Config)

cs.store(name="run", node=Run)
