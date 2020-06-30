# import unittest
# import pytest
# import os, sys
# import subprocess


# # Add the local folder to the import path:
# network_dir = os.path.dirname(os.path.abspath(__file__))
# network_dir = os.path.dirname(network_dir)
# network_dir = network_dir.rstrip("tests/")
# sys.path.insert(0,network_dir)



# def test_torch_default_network():

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



