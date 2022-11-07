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

class Norm(Enum):
    none  = 0
    batch = 1
    layer = 2

@dataclass
class Vertex:
    active:  bool = False
    depth:    int = 0
    weight: float = 10.0
    n_layers: int = 3

@dataclass
class EventLabel:
    active:    bool  = True
    weight:    float = 0.1
    n_filters: int   = 256
    n_layers:  int   = 3


@dataclass
class Network:
    name:                 str          = "default"
    bias:                 bool         = True
    normalization:        Norm         = Norm.none
    n_initial_filters:    int          = 16
    blocks_per_layer:     int          = 2
    blocks_deepest_layer: int          = 5
    blocks_final:         int          = 5
    depth:                int          = 7
    filter_size_deepest:  int          = 5
    bottleneck_deepest:   int          = 256
    residual:             bool         = True
    block_concat:         bool         = False
    weight_decay:         float        = 0.01
    connections:          Connection   = Connection.concat
    conv_mode:            ConvMode     = ConvMode.conv_2D
    growth_rate:          GrowthRate   = GrowthRate.additive
    downsampling:         DownSampling = DownSampling.max_pooling
    upsampling:           UpSampling   = UpSampling.interpolation
    data_format:          str          = MISSING
    classification:       EventLabel   = EventLabel()
    vertex:               Vertex       = Vertex()


@dataclass
class UResNet(Network):
    name:                 str          = "uresnet"
    normalization:        Norm         = Norm.none

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
    normalization:        Norm         =  Norm.none

@dataclass
class Polaris(Network):
    name:                 str          = "polaris"
    bias:                 bool         = True
    blocks_deepest_layer: int          = 2
    blocks_final:         int          = 2
    depth:                int          = 7
    bottleneck_deepest:   int          = 96
    residual:             bool         = False
    connections:          Connection   = Connection.sum

@dataclass
class UResNet3D(Network):
    name:                 str          = "uresnet3d"
    conv_mode:            ConvMode     = ConvMode.conv_3D



cs = ConfigStore.instance()
# cs.store(name="network", node=Network)
cs.store(group="network", name="uresnet",   node=UResNet)
cs.store(group="network", name="a21",       node=A21)
cs.store(group="network", name="scc21",     node=SCC21)
cs.store(group="network", name="polaris",   node=Polaris)
cs.store(group="network", name="uresnet3d", node=UResNet3D)
