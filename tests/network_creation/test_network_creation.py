import unittest
import pytest
import os, sys
import subprocess


# Add the local folder to the import path:
network_dir = os.path.dirname(os.path.abspath(__file__))
network_dir = os.path.dirname(network_dir)
network_dir = network_dir.rstrip("tests/")
sys.path.insert(0,network_dir)

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
@pytest.mark.parametrize('n_initial_filters', [1, 8])
@pytest.mark.parametrize('conv_mode', ['2D', '3D'])
def test_build_network(connections, residual, batch_norm, use_bias,
    blocks_deepest_layer, blocks_per_layer, network_depth, blocks_final, 
    downsampling, upsampling, data_format, n_initial_filters, conv_mode):
    


    # Using subprocess to spin up these runs.

    # Run the tensorflow version:
    command = ["python", f"{network_dir}/bin/exec.py"]

    command += ["build_net"]
    command += ["-i", "1"]
    command += ["--synthetic", "True"]
    command += ["-m", "CPU"]
    command += ["--downsample-images", "3"]



    command += ["--connections",  str(connections)]
    command += ["--residual",  str(residual)]
    command += ["--batch-norm",  str(batch_norm)]
    command += ["--use-bias",  str(use_bias)]
    command += ["--blocks-deepest-layer",  str(blocks_deepest_layer)]
    command += ["--blocks-per-layer",  str(blocks_per_layer)]
    command += ["--network-depth",  str(network_depth)]
    command += ["--blocks-final",  str(blocks_final)]
    command += ["--downsampling",  str(downsampling)]
    command += ["--upsampling",  str(upsampling)]
    command += ["--data-format",  str(data_format)]
    command += ["--n-initial-filters",  str(n_initial_filters)]
    command += ["--conv-mode", str(conv_mode)]

    print()
    print("Executed command: ")
    print(" ".join(command))
    print()


    completed_proc_tf = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if completed_proc_tf.returncode != 0:
        print(f"Torch return code: {completed_proc_tf}")
        print(completed_proc_tf.stdout)
        print(completed_proc_tf.stderr)
        assert False

    # Run the pytorch version:

    command += ["--framework", "torch"]
    completed_proc_torch = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


    if completed_proc_torch.returncode != 0:
        print(f"Torch return code: {completed_proc_torch}")
        print(completed_proc_torch.stdout)
        print(completed_proc_torch.stderr)
        assert False

    n_params_tf    = int(completed_proc_tf.stdout.decode().split()[-1])
    n_params_torch = int(completed_proc_torch.stdout.decode().split()[-1])

    if n_params_torch != n_params_tf:
        print()
        print("Executed command: ")
        print(" ".join(command))
        print()

    assert n_params_torch == n_params_tf

    # if completed_proc.returncode > 0:
    #     assert True
    # else:
    #     print(completed_proc.stdout)
    #     try:
    #         print(completed_proc.stderr)
    #     except:
    #         pass

