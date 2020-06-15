# import unittest
# import pytest
# import os, sys
# import subprocess


# # Add the local folder to the import path:
# network_dir = os.path.dirname(os.path.abspath(__file__))
# network_dir = os.path.dirname(network_dir)
# network_dir = network_dir.rstrip("tests/")
# sys.path.insert(0,network_dir)



# def test_torch_3D_default_network():

#     # Using subprocess to spin up these runs.

#     command = ["python", f"{network_dir}/bin/exec.py"]

#     command += ["train"]
#     command += ["--framework", "torch"]
#     command += ["-i", "1"]
#     command += ["--synthetic", "True"]
#     command += ["-m", "CPU"]
#     command += ["--downsample-images", "3"]
#     command += ["--network-depth", "3"]


#     completed_proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


#     if completed_proc.returncode == 0:
#         assert True
#     else:
#         print(command)
#         print(completed_proc.stdout)
#         try:
#             print(completed_proc.stderr)
#         except:
#             pass

#         assert False


# @pytest.mark.full
# @pytest.mark.parametrize('connections', ['sum', 'concat', 'none'])
# @pytest.mark.parametrize('residual', [True, False])
# @pytest.mark.parametrize('batch_norm', [ True, False])
# @pytest.mark.parametrize('use_bias', [ True, False])
# @pytest.mark.parametrize('blocks_deepest_layer', [2] )
# @pytest.mark.parametrize('blocks_per_layer', [2] )
# @pytest.mark.parametrize('network_depth', [2] )
# @pytest.mark.parametrize('blocks_final', [1] )
# @pytest.mark.parametrize('downsampling', ['convolutional', 'max_pooling'] )
# @pytest.mark.parametrize('upsampling', ['convolutional', 'interpolation'] )
# @pytest.mark.parametrize('data_format', ['channels_last', 'channels_first'] )
# @pytest.mark.parametrize('n_initial_filters', [1, 8])
# def test_torch_3D_build_network(connections, residual, batch_norm, use_bias,
#     blocks_deepest_layer, blocks_per_layer, network_depth, blocks_final, 
#     downsampling, upsampling, data_format, n_initial_filters):
    


#     # Using subprocess to spin up these runs.

#     command = ["python", f"{network_dir}/bin/exec.py"]

#     command += ["train"]
#     command += ["--framework", "torch"]
#     command += ["-i", "1"]
#     command += ["--synthetic", "True"]
#     command += ["-m", "CPU"]
#     command += ["--downsample-images", "3"]



#     command += ["--connections",  str(connections)]
#     command += ["--residual",  str(residual)]
#     command += ["--batch-norm",  str(batch_norm)]
#     command += ["--use-bias",  str(use_bias)]
#     command += ["--blocks-deepest-layer",  str(blocks_deepest_layer)]
#     command += ["--blocks-per-layer",  str(blocks_per_layer)]
#     command += ["--network-depth",  str(network_depth)]
#     command += ["--blocks-final",  str(blocks_final)]
#     command += ["--downsampling",  str(downsampling)]
#     command += ["--upsampling",  str(upsampling)]
#     command += ["--data-format",  str(data_format)]
#     command += ["--n-initial-filters",  str(n_initial_filters)]


#     completed_proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


#     if completed_proc.returncode == 0:
#         assert True
#     else:
#         print(completed_proc.stdout)
#         try:
#             print(completed_proc.stderr)
#         except:
#             pass

#         assert False


