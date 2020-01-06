import unittest
import pytest
import os, sys


# Add the local folder to the import path:
network_dir = os.path.dirname(os.path.abspath(__file__))
network_dir = os.path.dirname(network_dir)
network_dir = network_dir.rstrip("tests/")
sys.path.insert(0,network_dir)


from src.networks.torch import uresnet2D
from src.utils.core import flags


import torch



def test_torch_default_network():
    FLAGS = flags.uresnet()
    FLAGS._set_defaults()
    FLAGS.SYNTHETIC=True

    FLAGS.MODE = "CI"

    FLAGS.dump_config()

    from src.utils.torch import trainer
    trainer = trainer.torch_trainer()
    trainer._initialize_io()
    # trainer = trainercore.trainercore()
    trainer.init_network()


@pytest.mark.full
@pytest.mark.parametrize('connections', ['sum', 'concat', 'none'])
@pytest.mark.parametrize('residual', [True, False])
@pytest.mark.parametrize('batch_norm', [ True, False])
@pytest.mark.parametrize('use_bias', [ True, False])
@pytest.mark.parametrize('blocks_deepest_layer', [2] )
@pytest.mark.parametrize('blocks_per_layer', [2] )
@pytest.mark.parametrize('network_depth', [2] )
@pytest.mark.parametrize('blocks_final', [1] )
@pytest.mark.parametrize('downsampling', ['convolutional', 'max_pooling'] )
@pytest.mark.parametrize('upsampling', ['convolutional', 'interpolation'] )
@pytest.mark.parametrize('data_format', ['channels_last', 'channels_first'] )
@pytest.mark.parametrize('n_initial_filters', [1])
def test_torch_build_network(connections, residual, batch_norm, use_bias,
    blocks_deepest_layer, blocks_per_layer, network_depth, blocks_final, 
    downsampling, upsampling, data_format, n_initial_filters):
    FLAGS = flags.uresnet()
    FLAGS._set_defaults()
    FLAGS.SYNTHETIC=True

    FLAGS.CONNECTIONS          = connections
    FLAGS.RESIDUAL             = residual
    FLAGS.BATCH_NORM           = batch_norm
    FLAGS.USE_BIAS             = use_bias
    FLAGS.BLOCKS_DEEPEST_LAYER = blocks_deepest_layer
    FLAGS.BLOCKS_PER_LAYER     = blocks_per_layer
    FLAGS.NETWORK_DEPTH        = network_depth
    FLAGS.BLOCKS_FINAL         = blocks_final
    FLAGS.DOWNSAMPLING         = downsampling
    FLAGS.UPSAMPLING           = upsampling
    FLAGS.DATA_FORMAT          = data_format
    FLAGS.N_INITIAL_FILTERS    = n_initial_filters

    FLAGS.MODE = "CI"

    FLAGS.dump_config()

    from src.utils.torch import trainer
    trainer = trainer.torch_trainer()
    trainer._initialize_io()
    # trainer = trainercore.trainercore()
    trainer.init_network()



