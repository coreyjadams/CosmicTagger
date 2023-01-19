from mpi4py import MPI
import socket
import numpy
import sys, os

def mpi_init_and_local_rank(set_env=True, port=2345, verbose=False):
    '''
    Determine the local rank as well as set environment variables needed for
    distributed training, if set_env is true.

    Variables set:
    WORLD_SIZE  - total number of ranks (N), N = total devices
    RANK        - the global rank of this device (0 to N-1)
    LOCAL_RANK  - the rank of this device on it's host (0 to n), n = devices / node
    LOCAL_SIZE  - the total number of devices
    MASTER_ADDR - the address of rank 0
    MASTER_PORT - the port to use on rank 0
    NODE_RANK   - the index of this particular node (0 to N/n - 1)
    N_NODES     - the total number of unique nodes.
    '''
    # Get the global communicator:
    COMM_WORLD = MPI.COMM_WORLD

    # This script can, optionally, set results into environment variables:
    if set_env:
        os.environ['WORLD_SIZE'] = str(COMM_WORLD.Get_size())
        os.environ['RANK'] = str(COMM_WORLD.Get_rank())



    # The strategy here is to split into sub communicators
    # Each sub communicator will be just on a single host,
    # And that communicator will assign ranks that can be interpretted
    # as local ranks.

    # To subdivide, each host will need to use a unique key.
    # We'll rely on the hostname and order them all.

    hostname = socket.gethostname()
    # host_key = host_key %
    all_hostnames = COMM_WORLD.gather(hostname, root=0)

    if COMM_WORLD.Get_rank() == 0:
        # Order all the hostnames, and find unique ones
        unique_hosts = numpy.unique(all_hostnames)
        # Numpy automatically sorts them.
    else:
        unique_hosts = None

    # Broadcast the list of hostnames:
    unique_hosts = COMM_WORLD.bcast(unique_hosts, root=0)
    # Find the integer for this host in the list of hosts:
    i = int(numpy.where(unique_hosts == hostname)[0])
    # print(f"{hostname} found itself at index {i}")

    if set_env:
        os.environ['NODE_RANK'] = str(i)
        os.environ['N_NODES']   = str(len(unique_hosts))


    new_comm = COMM_WORLD.Split(color=i)
    if verbose:
        print(f"Global rank {COMM_WORLD.Get_rank()} of {COMM_WORLD.Get_size()} mapped to local rank {new_comm.Get_rank()} of {new_comm.Get_size()} on host {hostname}")


    if set_env:
        os.environ['LOCAL_SIZE'] = str(new_comm.Get_size())
        os.environ['LOCAL_RANK'] = str(new_comm.Get_rank())

        # It will want the master address too, which we'll broadcast:
        if COMM_WORLD.Get_rank() == 0:
            master_addr = socket.gethostname()
            sock = socket.socket()
            sock.bind(('',0))
            master_port  = sock.getsockname()[1]
            master_port  = port
        else:
            master_addr = None
            master_port = None
        master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)
        master_port = MPI.COMM_WORLD.bcast(master_port, root=0)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)



    # The rank in the new communicator - which is host-local only - IS the local rank:
    return new_comm.Get_rank()

if __name__ == "__main__":
    local_rank(verbose=True)
