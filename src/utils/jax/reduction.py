import jax
import jax.numpy as numpy

from jax import jit


import math


@jit
def unflatten_tensor_into_tree(inputs, shapes, treedef):

    # Use this to keep track of where the flat index is:
    running_index = 0;

    input_leaf_values = []

    for shape in shapes:
        # How big is the leaf?
        input_size = math.prod(shape)
        # Take a slice that size:
        input_slice = inputs[running_index:running_index+input_size]
        # Add it to the output tree values shaped properly:
        input_leaf_values.append(input_slice.reshape(shape))
        # Update the start point
        running_index += input_size

    return jax.tree_util.tree_unflatten(treedef, input_leaf_values)

@jit
def flatten_tree_into_tensor(input_tree):

    # Flatten the tree structure into a flat structure:
    leaf_values, treedef = jax.tree_util.tree_flatten(input_tree)

    # Extract the shapes of the tensors:
    shapes = [ t.shape for t in leaf_values ]

    # Flatten every tensor:
    flattened = [t.flatten() for t in leaf_values ]

    # Combine the tensors:
    flat_tensor = numpy.concatenate(flattened, axis=0)

    return flat_tensor, shapes, treedef


@jit
def allreduce_dict(dictionary):
    '''
    To Do - this could likely benefit from bucketing.
    '''
    from mpi4py import MPI
    import mpi4jax

    # First, we flatten the dictionary:
    flat_tensor, shapes, treedef = flatten_tree_into_tensor(dictionary)

    # Call MPI and perform the allreduce:
    flat_tensor, mpi_token = mpi4jax.allreduce(
        flat_tensor,
        op = MPI.SUM,
        comm = MPI.COMM_WORLD,
        token = None
    )

    # # Ensure synchronization!
    # # Include a barrier here before using the output, necessary for avoid race conditions
    # token = mpi4jax.barrier(
    #     comm = MPI.COMM_WORLD,
    #     token = mpi_token
    # )

    reduced_dict = unflatten_tensor_into_tree(flat_tensor, shapes, treedef)
    return reduced_dict



def create_reduction_op(args):

    if not args.run.distributed:

        def reduction(tree):
            return tree
        
        return reduction

    else:
        # This needs to be an all reduce:
        return allreduce_dict