import unittest
import pytest
import os, sys
import subprocess

# Add the local folder to the import path:
network_dir = os.path.dirname(os.path.abspath(__file__))
network_dir = os.path.dirname(network_dir)
network_dir = network_dir.rstrip("tests/")
sys.path.insert(0,network_dir)




@pytest.mark.parametrize('synthetic', [False, True])
@pytest.mark.parametrize('downsample_images', [1, 2])
def test_torch_default_network(tmpdir, synthetic, downsample_images):
    
    # Instead of calling the python objects, use subprocesses 

    # first, where is the exec.py?
    exec_script = network_dir + "/bin/exec.py"

    args = [exec_script, "mode=train"]
    args += ["framework=torch"]
    args += [f"data.synthetic={synthetic}"]

    file_path = network_dir + "/example_data/"

    if not synthetic:
        args += [f"data.data_directory={file_path}"]
        args += [f"data.file=cosmic_tagging_light.h5"]


    
    args += ["run.id=0"]
    args += ["run.iterations=5"]
    args += ["network.n_initial_filters=1"]
    args += [f"network.network_depth={6 - downsample_images}"]
    args += [f"data.downsample={downsample_images}"]
    args += ["run.compute_mode=CPU"]
    


    random_file_name = str(tmpdir + "/torch_log_dir/")
    args += [f"run.output_dir={random_file_name}"]
    print(args)

    completed_proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


    if completed_proc.returncode == 0:
        assert True
    else:
        print(completed_proc.stdout)
        try:
            print(completed_proc.stderr)
        except:
            pass

        assert False







# if __name__ == '__main__':
#     test_torch_default_network("./", synthetic=True, downsample_images=2)






