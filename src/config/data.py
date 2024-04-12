from enum import Enum

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from typing import Tuple, Any
from omegaconf import MISSING

class DataFormatKind(Enum):
    channels_last  = 0
    channels_first = 1

class RandomMode(Enum):
    random_blocks = 0
    serial_access = 1

# data_top="/lus/grand/projects/datascience/cadams/datasets/SBND/cosmic_tagging_2/"
data_top="/lus/gila/projects/Aurora_deployment/cadams/cosmic_tagger_2/"
# data_top="/lus/gecko/projects/Aurora_deployment/cadams/cosmic_tagging_2/"

@dataclass
class DatasetPaths:
    train: str  = f"{data_top}/cosmic_tagging_2_train.h5"
    test:  str  = f"{data_top}/cosmic_tagging_2_test.h5"
    val:   str  = f"{data_top}/cosmic_tagging_2_val.h5"
    active: Tuple[str] =  field(default_factory=list)


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
    random_mode:      RandomMode = RandomMode.random_blocks
    img_transform:          bool = False
    seed:                    int = -1 # Random number seed
    paths:          DatasetPaths = field(default_factory = lambda : DatasetPaths() )


@dataclass
class Synthetic(Data):
    synthetic: bool = True


cs = ConfigStore.instance()
cs.store(group="data", name="real", node=Real)
# cs.store(group="data", name="val", node=Val)
# cs.store(group="data", name="test", node=Test)
cs.store(group="data", name="synthetic", node=Synthetic)
