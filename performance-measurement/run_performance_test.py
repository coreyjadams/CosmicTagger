import sys, os
import subprocess
import shutil
import time
from pathlib import Path


from enum import Enum


from common import get_parser, Framework, System, make_run_id, get_affinity

# This script wraps the cosmic tagger performance measurement into a comprehensive test suite.



def get_env_variables(system):
    
    env_update = {}
    if system == System.sunspot:

        env_update["NUMEXPR_MAX_THREADS"] = "1"
        env_update["ITEX_FP32_MATH_MODE"] = "TF32"
        env_update["ITEX_LAYOUT_OPT"] = ""
        env_update["FI_CXI_DEFAULT_CQ_SIZE"] = "131072"
        env_update["FI_CXI_OVFLOW_BUF_SIZE"] = "8388608"
        env_update["FI_CXI_CQ_FILL_PERCENT"] = "20"
        env_update["NUMEXPR_MAX_THREADS"] = "1"

    return env_update



def build_python_arguments(workdir, framework, system, precision, batch_size, iterations, config="a21"):

    # Configure the python args:
    python_args  = f"{workdir}/bin/exec.py --config-name {config} "
    python_args += f" data=synthetic "

    # Use this to make the checkpoint iteration in the far future:
    python_args += f" mode.checkpoint_iteration=20000"

    # Select Framework:
    if "pt" in framework.name:
        python_args += f" framework=torch"
    elif "tf" in framework.name:
        python_args += f" framework=tensorflow"

    # Select Distributed or not?
    if "ddp" in framework.name:

        python_args += f" run.distributed=True "
        python_args += f" framework.distributed_mode=DDP "

    elif "hvd" in framework.name:

        python_args += f" run.distributed=True "
        if "pt" in framework.name:
            python_args += f" framework.distributed_mode=horovod "
    else:
        python_args += f" run.distributed=False "

    if system == System.sunspot:
        python_args += f" run.compute_mode=XPU "
        python_args += f" data.data_format=channels_last "
        python_args += f" run.precision={precision} "
        python_args += f" run.minibatch_size={batch_size} "
        python_args += f" run.iterations={iterations}"

    return python_args

def get_hosts(system, work_dir):

    # Launch one process per node via mpi, that prints `hostname`, and collect the hosts.

    # # build up an mpi call:
    # proc_ags = ["mpiexec", "-n", n_procs, "-ppn", 1, "hostname"]

    # output = subprocess.run(proc_args, capture_output=True)
    # print(output)
    host_dest = work_dir / Path("hosts.txt")

    if system == System.sunspot or system == System.polaris or system == System.aurora:
        # CP the host file to the work_dir:
        shutil.copyfile(os.environ["PBS_NODEFILE"], host_dest)

    # Read in the hosts:
    with open(host_dest, 'r') as _f:
        hosts = _f.readlines()

    # remove new lines:
    hosts = [h.rstrip('\n') for h in hosts]

    return hosts

def ranks_per_node(system):

    if system == System.sunspot: return 12
    elif system == System.aurora: return 12

def run_single_node_benchmarks(args, framework, env_vars, affinity, python_args, dir, hosts):

    # This function makes a small script to run the single-node benchmark, doing collectives
    # over the whole node.

    
    # This function builds a small script to launch with mpirun.

    rpn = ranks_per_node(args.system)

    env_setup = ""
    for key in env_vars.keys():
        if env_vars[key] == "":
            env_setup += f"unset {key}\n"
        else:
            env_setup += f"export {key}={env_vars[key]}\n"


    run_id  = make_run_id(args, framework, hosts)

   
    # Have to massage the affinities into shell-script compatible strs:
    cpu_affinity =":".join(affinity["cpu_affinity"])

    # a list of processes to track, one per node:
    procs = []

    for host in hosts:
        host = host.split(".")[0]
        print(host)

        this_single_node_dir = dir / Path(run_id) / Path(host)
        this_single_node_dir.mkdir(exist_ok=True, parents=True)
        print(this_single_node_dir)

        script_template = """#!/bin/bash -l

# Set env variables:
{env_setup}

echo "Launching job from $(hostname) to {host}"

run_id="{run_id}-{host}-fullnode/"

mkdir -p {output_dir}
out_file={output_dir}/bash_log.txt 
touch ${{out_file}}

mpiexec -n {rpn} -ppn {rpn} \
--hosts={host} \
--cpu-bind=verbose,list:{cpu_affinity} \
python {python_args} \
output_dir={output_dir} \
run.id=${run_id} > ${{out_file}} 2>&1 

        """.format(
            run_id      = run_id,
            env_setup   = env_setup, 
            rpn         = rpn,
            output_dir  = this_single_node_dir,
            host        = host,
            python_args = python_args,
            cpu_affinity = cpu_affinity,
        )


        # print(script_template)

        script_path = this_single_node_dir / Path("single-node.sh")
        # print(script_path)
        # Write the script to a file
        with open(script_path, 'w') as f:
            f.write(script_template)

        # Make the script executable:
        subprocess.run(['chmod', 'u+x', str(script_path) ])


        # Launch the script with a POpen and don't wait ...     
        procs.append(subprocess.Popen([str(script_path)], ))
    

    while len(procs) > 0:
        print(procs)
        for proc in procs:
            if proc.poll() is None:
                continue
            else:
                procs.remove(proc)
        time.sleep(5)
    return


def run_multi_node_benchmarks(args, framework, env_vars, affinity, python_args, dir, hosts):

    # This function makes a small script to run the single-node benchmark, doing collectives
    # over the whole node.

    
    # This function builds a small script to launch with mpirun.

    rpn = ranks_per_node(args.system)

    env_setup = ""
    for key in env_vars.keys():
        if env_vars[key] == "":
            env_setup += f"unset {key}\n"
        else:
            env_setup += f"export {key}={env_vars[key]}\n"


    run_id  = make_run_id(args, framework, hosts)

   
    # Have to massage the affinities into shell-script compatible strs:
    cpu_affinity =":".join(affinity["cpu_affinity"])

    multinode_dir = dir / Path(run_id) 
    multinode_dir.mkdir(exist_ok=True, parents=True)
    print(multinode_dir)

    n_hosts = len(hosts)
    hosts = [ h.split(".")[0] for h in hosts]
    hosts = ",".join(hosts)
    print(hosts)

    script_template = """#!/bin/bash -l

# Set env variables:
{env_setup}

echo "Launching job from $(hostname) to all nodes"

run_id="{run_id}-multinode/"

mkdir -p {output_dir}
out_file={output_dir}/bash_log.txt 
touch ${{out_file}}

mpiexec -n {n_nodes} -ppn {rpn} \
--hosts={hosts} \
--cpu-bind=verbose,list:{cpu_affinity} \
python {python_args} \
output_dir={output_dir} \
run.id=${run_id} > ${{out_file}} 2>&1 

    """.format(
        run_id      = run_id,
        env_setup   = env_setup, 
        n_nodes     = rpn * n_hosts,
        rpn         = rpn,
        output_dir  = multinode_dir,
        hosts       = hosts,
        python_args = python_args,
        cpu_affinity = cpu_affinity,
    )


    # print(script_template)

    script_path = multinode_dir    / Path("multi-node.sh")
    # print(script_path)
    # Write the script to a file
    with open(script_path, 'w') as f:
        f.write(script_template)

    # Make the script executable:
    subprocess.run(['chmod', 'u+x', str(script_path) ])


    # Launch the script with a POpen and don't wait ...     
    proc = subprocess.Popen([str(script_path)], )

    proc.wait()

    return


def run_single_tile_benchmarks(args, framework, env_vars, affinity, python_args, dir, hosts):

    # This function builds a small script to launch with mpirun.

    rpn = ranks_per_node(args.system)

    env_setup = ""
    for key in env_vars.keys():
        if env_vars[key] == "":
            env_setup += f"unset {key}\n"
        else:
            env_setup += f"export {key}={env_vars[key]}\n"


    run_id  = make_run_id(args, framework, hosts)
    
    this_single_tile_dir = dir / Path(run_id)
    this_single_tile_dir.mkdir(exist_ok=True, parents=True)

    # Have to massage the affinities into shell-script compatible strs:
    gpu_affinity =" ".join(affinity["gpu_affinity"])
    cpu_affinity =" ".join(affinity["cpu_affinity"])

    script_template = """#!/bin/bash -l

# Set env variables:
{env_setup}

GPU_AFFINITY_LIST=({gpu_affinity})
CPU_AFFINITY_LIST=({cpu_affinity})

echo "Launching jobs on $(hostname)"

for (( idx=0; idx<{rpn}; idx++ ));
do
    GPU=${{GPU_AFFINITY_LIST[$idx]}}
    CPU=${{CPU_AFFINITY_LIST[$idx]}}
    host=$(hostname)
    run_id="{run_id}/${{host}}/GPU${{GPU}}"
    mkdir -p {output_dir}/${{run_id}}
    # echo ${{run_id}}
    export {gpu_variable}=${{GPU}}
    # echo ${{ZE_AFFINITY_MASK}}
    # echo ${{CPU}}
    out_file={output_dir}/${{run_id}}/bash_log.txt 
    touch ${{out_file}}
    numactl -C ${{CPU}} python {python_args} run.id=${{run_id}} output_dir={output_dir}/${{run_id}} > ${{out_file}} 2>&1 &
    unset {gpu_variable}
done
wait < <(jobs -p)

""".format(
    run_id      = run_id,
    env_setup   = env_setup, 
    rpn         = rpn,
    output_dir  = dir,
    python_args = python_args,
    gpu_affinity = gpu_affinity,
    cpu_affinity = cpu_affinity,
    gpu_variable = affinity["gpu_variable"],
)


    # print(script_template)

    script_path = this_single_tile_dir / Path("single-node.sh")
    # Write the script to a file
    with open(script_path, 'w') as f:
        f.write(script_template)

    # Make the script executable:
    subprocess.run(['chmod', 'u+x', str(script_path) ])


    # Build up the MPI call:

    mpi_args = ["mpiexec", "-n", len(hosts), "-ppn", 1, "--cpu-bind=none", script_path]
    proc = subprocess.Popen([ str(s) for s in mpi_args ])
    print(proc)
    proc.wait()

    return

def check_system_supported(system):

    if system == System.sunspot: return True
    if system == System.aurora: return True

    return False

def run_main_benchmark(args):

    # Infer the work directory for CosmicTagger based on the location of this script:
    cwd = os.getcwd()
    CT_DIR = os.path.dirname(cwd)
    print(CT_DIR)

    env_vars = get_env_variables(args.system)
    affinity = get_affinity(args.system)

    # First, prepare the output directory:
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # now, make a dir for the host list check:
    hosts_dir = args.out_dir / Path("hosts/")
    hosts_dir.mkdir(parents=True, exist_ok=True)

    hosts = get_hosts(args.system, hosts_dir)

    # Next, run the single-tile benchmarks

    for framework in [Framework.pt, Framework.tf]:

        # Build the python arguments:
        python_args = build_python_arguments(
            workdir    = CT_DIR, 
            framework  = framework, 
            system     = args.system, 
            precision  = args.precision, 
            batch_size = args.batch_size, 
            iterations = args.iterations,
            config     = "a21"
        )


        single_tile_dir = args.out_dir / Path("single-tile/")
        run_single_tile_benchmarks(args, framework, env_vars, affinity, python_args, single_tile_dir, hosts)


    for framework in [Framework.ptddp, Framework.pthvd, Framework.tfhvd]:

        # Build the python arguments:
        python_args = build_python_arguments(
            workdir    = CT_DIR, 
            framework  = framework, 
            system     = args.system, 
            precision  = args.precision, 
            batch_size = args.batch_size, 
            iterations = args.iterations,
            config     = "a21"
        )


        single_node_dir = args.out_dir / Path("single-node/")
        run_single_node_benchmarks(args, framework, env_vars, affinity, python_args, single_node_dir, hosts)

    for framework in [Framework.ptddp, Framework.pthvd, Framework.tfhvd]:

        # Build the python arguments:
        python_args = build_python_arguments(
            workdir    = CT_DIR, 
            framework  = framework, 
            system     = args.system, 
            precision  = args.precision, 
            batch_size = args.batch_size, 
            iterations = args.iterations,
            config     = "a21"
        )


        multi_node_dir = args.out_dir / Path("multi-node/")
        run_multi_node_benchmarks(args, framework, env_vars, affinity, python_args, multi_node_dir, hosts)




def main():

    parser = get_parser()

    args = parser.parse_args()

    # Convert the enums to the enum types:
    args.system = System[args.system.lower()]


    if not check_system_supported(args.system):
        raise Exception(f"{args.system.name}")

    print(args)


    run_main_benchmark(args)




    pass

if __name__ == "__main__":
    main()
