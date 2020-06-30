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
def test_tensorflow_default_network(tmpdir, synthetic, downsample_images):
    
    # Instead of calling the python objects, use subprocesses 

    # first, where is the exec.py?
    exec_script = network_dir + "/bin/exec.py"

    file_path = network_dir + "/example_data/"
    file_path += "cosmic_tagging_light.h5"

    args = [exec_script, "train"]
    args += ["--framework", "tensorflow"]
    args += ["--synthetic", f"{synthetic}"]

    if not synthetic:
        args += ["--file", f"{file_path}"]


    args += ["--iterations",  "5"]
    args += ["--n-initial-filters",  "1"]
    args += ["--network-depth",  "{}".format(6 - downsample_images)]
    args += ["--downsample-images",  f"{downsample_images}"]
    


    random_file_name = str(tmpdir + "/tensorflow_log_dir/")
    args += ["--log-directory", random_file_name]
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


# def test_tensorflow_model_inference(tmpdir):
    

#     # Instead of calling the python objects, use subprocesses 

#     # first, where is the exec.py?
#     exec_script = network_dir + "/bin/exec.py"

#     file_path = network_dir + "/example_data/"
#     file_path += "cosmic_tagging_light.h5"

#     args = [exec_script, "train"]
#     args += ["--framework", "tensorflow"]

#     args += ["--file", f"{file_path}"]

#     args += ["--iterations",  "5"]
#     args += ["--n-initial-filters",  "1"]
#     args += ["--network-depth",  "4"]
#     args += ["--downsample-images",  "2"]
    


#     random_file_name = str(tmpdir + "/tensorflow_log_dir/")
#     args += ["--log-directory", random_file_name]

#     completed_proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


#     if completed_proc.returncode != 0:
#         print(completed_proc.stdout)
#         try:
#             print(completed_proc.stderr)
#         except:
#             pass

#         assert False

#     # Now, reload the model and run inference:


#     args = [exec_script, "inference"]
#     args += ["--framework", "tensorflow"]


#     args += ["--iterations",  "5"]
#     args += ["--n-initial-filters",  "1"]
#     args += ["--network-depth",  "4"]
#     args += ["--downsample-images",  "2"]
#     args += ["--file", f"{file_path}"]


#     random_file_name = str(tmpdir + "/tensorflow_log_dir/")
#     args += ["--log-directory", random_file_name]

#     completed_proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    


#     if completed_proc.returncode == 0:
#         assert True
#     else:
#         print(completed_proc.stdout)
#         try:
#             print(completed_proc.stderr)
#         except:
#             pass

#         assert False

# if __name__ == '__main__':
#     test_tensorflow_default_network("./", synthetic=True, downsample_images=2)






