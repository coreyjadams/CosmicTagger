import sys, os
import subprocess

import time

# This script will matrix out a few different variations of the same script.
# Fill in the templates below:

batch_sizes = [1, 2, 4]
precisions  = ["float32", "mixed", "bfloat16"]

# Env variables are a dictionary of dictionaries.  The outer dictionary
# keys are the configuration names.
# The inner dictionaries get added to the env.


config_name = "a21"

core_command = [
    "python",
    "bin/exec.py",
    "--config-name",
    config_name,
    "data=synthetic",
    "run.iterations=100",
]

GPUS_AVAILABLE = ["0", "1", "2", "3"]

base_environ = os.environ

command_list = []
env_list = []
run_id = 0
for batch_size in [1, 2, 4]:
    for precision in ["float32", "mixed"]:
        for XLA in [True, False]:
            for TF32 in [True, False]:
                # Configure here:
                this_command = core_command + [f"run.minibatch_size={batch_size}",]
                this_command += [f"run.precision={precision}",]
                this_env = {}
                if XLA:
                    this_env.update({ "TF_XLA_FLAGS" : "--tf_xla_auto_jit=2"})
                if not TF32:
                    this_env.update({ "NVIDIA_TF32_OVERRIDE" : "0"})


                id_string = f"{run_id}_mb{batch_size}_{precision}"
                if XLA:
                    id_string += "_XLA"
                if not TF32:
                    id_string += "_noTF32"

                this_command += [f"run.id={id_string}"]
                # print(id_string)
                # print("  " + " ".join(this_command))
                # print(this_env)
                command_list.append(this_command)
                env_list.append(this_env)
                run_id += 1

print(len(command_list))
# Now, we loop through the command lists and assign a command to each GPU.
# When one terminates, another is spawned.
active_processes = []
active_gpus      = []
while len(active_processes) > 0 or len(command_list) > 0:
    # This while loop looks to make sure there are running processes
    # OR there are things less to run.

    # print("len(command_list): ", len(command_list))
    # print("len(active_processes): ", len(active_processes))
    # print("len(GPUS_AVAILABLE): ", len(GPUS_AVAILABLE))
    # Launch a new process, if possible:
    if len(command_list) > 0 and len(GPUS_AVAILABLE) > 0:
        # Get the first command and env:
        this_command = command_list[0]
        this_env     = env_list[0]
        this_gpu     = GPUS_AVAILABLE[0]

        # Specify this GPU:
        this_env.update({"CUDA_VISIBLE_DEVICES" : this_gpu})
        print("Opening a process")
        print(this_env)
        this_env.update(os.environ)
        proc = subprocess.Popen(
            this_command,
            env = this_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT)

        active_processes.append(proc)
        active_gpus.append(this_gpu)

        # Pop the commands off the list:
        command_list.pop(0)
        env_list.pop(0)
        GPUS_AVAILABLE.pop(0)
    # else:
    #     print("Not spawing a process")

    print(active_processes)
    # print(active_gpus)
    # Check all active scripts:
    for i, proc in enumerate(active_processes):
        if proc.poll() is None:
            continue
        else:
            # This job is done
            print("Done a job")
            # Return the GPU to the pool:
            GPUS_AVAILABLE.append(active_gpus[i])
            # Remove these items from the tracking list
            active_gpus.pop(i)
            active_processes.pop(i)

    # Don't go crazy, sleep a second or two:
    time.sleep(5)
