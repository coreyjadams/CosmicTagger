from enum import Enum

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore


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
    2D: 0
    3D: 1


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
    conv_mode:            ConvMode     = ConvMode.2D
    connections:          Connection   = Connection.concat
    growth_rate:          GrowthRate   = GrowthRate.additive
    downsampling:         DownSampling = DownSampling.max_pooling
    upsampling:           UpSampling   = UpSampling.interpolation
    data_format:          str          = ${data.data_format}

cs = ConfigStore.instance()
cs.store(name="network", node=Network)