from enum import Enum

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

class DataFormatKind(Enum):
    channels_last  = 0
    channels_first = 1

@dataclass
class Data:
    synthetic:              bool = False
    downsample:              int = 1
    data_format: DataFormatKind  = DataFormatKind.channels_last
    img_transform:          bool = False
    version:                 int = 2 # Pick 1 or two
    seed:                    int = 0

@dataclass
class Real(Data):
    synthetic:      bool = False
    data_directory: str  = "/lus/grand/projects/datascience/cadams/datasets/SBND/cosmic_tagging_2/"
    file:           str  = "cosmic_tagging_2_train.h5"
    aux_file:       str  = "cosmic_tagging_2_val.h5"




@dataclass
class Synthetic(Data):
    synthetic: bool = True


cs = ConfigStore.instance()
cs.store(group="data", name="real", node=Real)
cs.store(group="data", name="synthetic", node=Synthetic)
