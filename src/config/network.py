from enum import Enum

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


# @dataclass
class Connection(Enum):
    concat = 0
    sum    = 1
    none   = 2

class GrowthRate(Enum):
    multiplicative = 0
    additive       = 1

class DownSampling(Enum):
    convolutional = 0
    max_pooling   = 1

class UpSampling(Enum):
    convolutional = 0
    interpolation = 1


class ConvMode(Enum):
    conv_2D = 0
    conv_3D = 1


@dataclass 
class Network:
    name:                 str          = "default"
    bias:                 bool         = True
    batch_norm:           bool         = True
    n_initial_filters:    int          = 16
    blocks_per_layer:     int          = 2
    blocks_deepest_layer: int          = 5
    blocks_final:         int          = 5
    network_depth:        int          = 6
    filter_size_deepest:  int          = 5
    bottleneck_deepest:   int          = 256
    residual:             bool         = True
    block_concat:         bool         = False
    weight_decay:         float        = 0.0
    connections:          Connection   = Connection.concat
    conv_mode:            ConvMode     = ConvMode.conv_2D
    growth_rate:          GrowthRate   = GrowthRate.additive
    downsampling:         DownSampling = DownSampling.max_pooling
    upsampling:           UpSampling   = UpSampling.interpolation
    data_format:          str          = MISSING

@dataclass
class UResNet(Network):
    name:                 str          = "uresnet"


@dataclass 
class A21(Network):
    name:                 str          = "A21"
    n_initial_filters:    int          = 8
    filter_size_deepest:  int          = 5
    residual:             bool         = False
    block_concat:         bool         = False
    growth_rate:          GrowthRate   = GrowthRate.additive
    data_format:          str          = MISSING

@dataclass
class SCC21(Network):
    name:                 str          = "scc21"
    batch_norm:           bool         = False

@dataclass
class Polaris(Network):
    name:                 str          = "polaris"
    bias:                 bool         = True
    blocks_deepest_layer: int          = 2
    blocks_final:         int          = 2
    network_depth:        int          = 7
    bottleneck_deepest:   int          = 96
    residual:             bool         = False
    connections:          Connection   = Connection.sum




cs = ConfigStore.instance()
# cs.store(name="network", node=Network)
cs.store(group="network", name="uresnet", node=UResNet)
cs.store(group="network", name="a21",     node=A21)
cs.store(group="network", name="scc21",   node=SCC21)
cs.store(group="network", name="polaris", node=Polaris)