import unittest
import pytest
import os, sys


# Add the local folder to the import path:
network_dir = os.path.dirname(os.path.abspath(__file__))
network_dir = os.path.dirname(network_dir)
network_dir = network_dir.rstrip("tests/")
sys.path.insert(0,network_dir)

from src.utils.core import flags

@pytest.mark.io
def test_io():
    
    from bin.exec import main


    file_path = network_dir + "/example_data/"
    file_path += "cosmic_tagging_light.h5"

    # flags is a singleton, meaning we can edit here and it will pick up changes
    FLAGS = flags.uresnet()
    FLAGS._set_defaults()
    FLAGS.FILE = file_path
    FLAGS.MODE = 'iotest'
    FLAGS.ITERATIONS = 2

    main(skip_parsing = True)



