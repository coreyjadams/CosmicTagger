import unittest
import pytest
import os, sys


# Add the local folder to the import path:
network_dir = os.path.dirname(os.path.abspath(__file__))
network_dir = os.path.dirname(network_dir)
network_dir = network_dir.rstrip("test/io/")
sys.path.insert(0,network_dir)


from src.networks.torch import uresnet2D
from src.utils.core import flags


import torch

@pytest.mark.io
def test_io():
    
    from bin.exec import main


    file_path = network_dir + "/example_data/"
    file_path += "cosmic_tagging_dev.h5"

    # flags is a singleton, meaning we can edit here and it will pick up changes
    FLAGS = flags.uresnet()
    FLAGS._set_defaults()
    FLAGS.FILE = file_path
    FLAGS.MODE = 'iotest'
    FLAGS.ITERATIONS = 10

    main(skip_parsing = True)



