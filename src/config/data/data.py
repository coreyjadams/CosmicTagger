from enum import Enum

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

@dataclass
class Data:
    synthetic:   bool = MISSING
    downsample:  int  = 1
    data_format: str  = MISSING

@dataclass
class Real(Data):
    synthetic:      bool = False
    data_directory: str  = MISSING
    file:           str  = "cosmic_tagging_train.h5"
    aux_file:       str  = "cosmic_tagging_test.h5"

@dataclass
class Synthetic(Data):
    synthetic: bool = True


cs = ConfigStore.instance()
cs.store(group="data", name="real", node=Real)
cs.store(group="data", name="synthetic", node=Synthetic)