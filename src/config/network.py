from enum import Enum

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from typing import List, Any


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
    group = 3
    instance = 4

class BlockStyle(Enum):
    none    = 0
    residual = 1
    convnext = 2

@dataclass
class Vertex:
    detach:   bool = True
    active:   bool = False
    depth:     int = 2
    weight:  float = 1.0
    n_layers:  int = 4
    n_filters: int = 128
    l_det:   float = 1e-1
    l_coord: float = 1e-1

@dataclass
class EventLabel:
    detach:   bool = True
    active:   bool = False
    weight:  float = 0.01
    n_filters: int = 128
    n_layers:  int = 4

@dataclass
class Backbone:
    name:           str        = ""
    vertex:         Vertex     = field(default_factory= lambda : Vertex() )
    classification: EventLabel = field(default_factory= lambda : EventLabel() )
    activation:     str        = "leaky_relu"
    data_format:    str        = MISSING


@dataclass
class Segformer(Backbone):
    name: str = "segformer"
    in_dims: List[int] = field(default_factory= lambda : [32, 64, 160, 256])
    decoder_dim: int =256

@dataclass
class CvT(Backbone):
    name:  str = "cvt"
    layer_embed: List[int] = field(default_factory= lambda : [ 32, 64, 64])
    blocks:      List[int] = field(default_factory= lambda : [ 2, 2, 2])
    num_heads:   List[int] = field(default_factory= lambda : [ 2, 8, 8])

@dataclass
class ConvNetwork(Backbone):
    name:                 str          = "default"
    bias:                 bool         = True
    normalization:        Norm         = Norm.batch
    n_initial_filters:    int          = 16
    blocks_per_layer:     int          = 2
    blocks_deepest_layer: int          = 5
    blocks_final:         int          = 5
    depth:                int          = 7
    kernel_size:          int          = 3
    bottleneck_deepest:   int          = 256
    depthwise:            bool         = False
    block_style:          BlockStyle   = BlockStyle.residual
    block_concat:         bool         = False
    weight_decay:         float        = 0.00
    connections:          Connection   = Connection.concat
    conv_mode:            ConvMode     = ConvMode.conv_2D
    growth_rate:          GrowthRate   = GrowthRate.additive
    downsampling:         DownSampling = DownSampling.max_pooling
    upsampling:           UpSampling   = UpSampling.interpolation


@dataclass
class UResNet(ConvNetwork):
    name:                 str          = "uresnet"
    normalization:        Norm         = Norm.layer

@dataclass
class A21(ConvNetwork):
    """
    In tensorflow, this model should have 8516083 parameters total.
    In pytorch, this model should have 8510547 parameters total.
    """
    name:                 str          = "A21"
    n_initial_filters:    int          = 8
    kernel_size:          int          = 3
    block_style:          BlockStyle   = BlockStyle.none
    block_concat:         bool         = False
    blocks_final:         int          = 0
    growth_rate:          GrowthRate   = GrowthRate.additive
    data_format:          str          = MISSING
    connections:          Connection   = Connection.sum
    normalization:        Norm         = Norm.batch
    depth:                int          = 6

@dataclass
class SCC21(ConvNetwork):
    name:                 str          = "scc21"
    normalization:        Norm         =  Norm.none

@dataclass
class Polaris(ConvNetwork):
    name:                 str          = "polaris"
    bias:                 bool         = True
    blocks_deepest_layer: int          = 2
    blocks_final:         int          = 2
    depth:                int          = 7
    bottleneck_deepest:   int          = 96
    block_style:          BlockStyle   = BlockStyle.none
    connections:          Connection   = Connection.sum

@dataclass
class UResNet3D(ConvNetwork):
    name:                 str          = "uresnet3d"
    conv_mode:            ConvMode     = ConvMode.conv_3D



cs = ConfigStore.instance()
# cs.store(name="network", node=Network)
cs.store(group="network", name="uresnet",   node=UResNet)
cs.store(group="network", name="a21",       node=A21)
cs.store(group="network", name="scc21",     node=SCC21)
cs.store(group="network", name="polaris",   node=Polaris)
cs.store(group="network", name="uresnet3d", node=UResNet3D)
cs.store(group="network", name="segformer", node=Segformer)
cs.store(group="network", name="cvt",       node=CvT)
