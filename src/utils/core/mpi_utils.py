from mpi4py import MPI
import socket
import numpy

def local_rank(verbose=False):

    # Get the global communicator:
    COMM_WORLD = MPI.COMM_WORLD

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

    new_comm = COMM_WORLD.Split(color=i)
    if verbose:
        print(f"Global rank {COMM_WORLD.Get_rank()} of {COMM_WORLD.Get_size()} mapped to local rank {new_comm.Get_rank()} of {new_comm.Get_size()} on host {hostname}")

    # The rank in the new communicator - which is host-local only - IS the local rank:
    return new_comm.Get_rank()

if __name__ == "__main__":
    local_rank(verbose=True)
