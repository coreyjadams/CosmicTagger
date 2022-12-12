from enum import Enum

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from typing import List, Any
from omegaconf import MISSING

from .network   import Network
from .mode      import Mode
from .framework import Framework
from .data      import Data

class ComputeMode(Enum):
    CPU   = 0
    GPU   = 1
    DPCPP = 2
    XPU   = 3

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
    iterations:         int         = 500
    epoch:              int         = 1000
    aux_iterations:     int         = 10
    minibatch_size:     int         = 2
    # aux_minibatch_size: int         = MISSING
    id:                 str         = MISSING
    precision:          Precision   = Precision.float32
    profile:            bool        = False

cs = ConfigStore.instance()

cs.store(group="run", name="base_run", node=Run)

cs.store(
    name="disable_hydra_logging",
    group="hydra/job_logging",
    node={"version": 1, "disable_existing_loggers": False, "root": {"handlers": []}},
)


defaults = [
    {"run"       : "base_run"},
    {"mode"      : "train"},
    {"data"      : "real"},
    {"framework" : "tensorflow"},
    {"network"   : "uresnet"}
]

@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    #         "_self_",
    #         {"run" : Run()}
    #     ]
    # )

    run:        Run       = MISSING
    mode:       Mode      = MISSING
    data:       Any       = MISSING
    framework:  Framework = MISSING
    network:    Network   = MISSING
    output_dir: str       = "output/"


cs.store(name="base_config", node=Config)
