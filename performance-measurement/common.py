from pathlib import Path
from enum import Enum


class System(Enum):

    polaris = 0
    sunspot = 1
    aurora  = 2

class Framework(Enum):

    pt     = 0
    ptddp  = 1
    pthvd  = 2
    tf     = 3
    tfhvd  = 4


def make_run_id(args, framework,  hosts):

    n_nodes = len(hosts) 
    run_id = f"{args.system.name}_{framework.name}_"
    run_id += f"bs{args.batch_size}_i{args.iterations}_"
    run_id += f"n{n_nodes}_p{args.precision}"



    return run_id


def get_affinity(system):

    if system == System.sunspot or system == System.aurora:
        gpu_affinity = ["0.0","0.1","1.0","1.1","2.0","2.1",
                        "3.0","3.1","4.0","4.1","5.0","5.1"]
        cpu_affinity = ["0-7,104-111",   "8-15,112-119",  "16-23,120-127", 
                        "24-31,128-135", "32-39,136-143", "40-47,144-151", 
                        "52-59,156-163", "60-67,164-171", "68-75,172-179", 
                        "76-83,180-187", "84-91,188-195", "92-99,196-203",]
        gpu_variable="ZE_AFFINITY_MASK"
        cpu_type="list"

    
    return {
        "gpu_variable" : gpu_variable,
        # "cpu_type"     : cpu_type,
        "gpu_affinity" : gpu_affinity, 
        "cpu_affinity" : cpu_affinity, 
    }

    pass


def get_parser():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", "-s", type=str,
                        help = "Which supercomputing system to run on",
                        choices = System.__members__.keys(),
                        required = True)

    parser.add_argument("--iterations", "-i", type=int,
                        help = "How many iterations to use for each run (recommend 100 or more)",
                        required = False, default = 102)

    parser.add_argument("--batch_size", "-bs", type=int,
                        help = "Local batch size per rank",
                        required = False, default = 8)

    parser.add_argument("--precision", "-p", type=str,
                        help = "Which precision to use to compute",
                        required = False, default = "float32",
                        choices = ["float32", "bfloat16", "mixed"])

    parser.add_argument("--out-dir", "-o", type=Path,
                        help = "Output directory to write to",
                        required = True)

    return parser
