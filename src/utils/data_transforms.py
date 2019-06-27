import numpy

'''
This is a torch-free file that exists to massage data
From sparse to dense or dense to sparse, etc.

This can also convert from sparse to sparse to rearrange formats
For example, larcv BatchFillerSparseTensor2D (and 3D) output data
with the format of 
    [B, N_planes, Max_voxels, N_features] 

where N_features is 2 or 3 depending on whether or not values are included 
(or 3 or 4 in the 3D case)

# The input of a pointnet type format can work with this, but SparseConvNet
# requires a tuple of (coords, features, [batch_size, optional])


''' 

def larcvsparse_to_dense_2d(input_array, dense_shape):

    batch_size = input_array.shape[0]
    n_planes   = input_array.shape[1]
    output_array = numpy.zeros((batch_size, n_planes, dense_shape[0], dense_shape[1]), dtype=numpy.float32)


    x_coords = input_array[:,:,:,1]
    y_coords = input_array[:,:,:,0]
    val_coords = input_array[:,:,:,2]

    # print(input_array.shape)


    # Find the non_zero indexes of the input:
    batch_index, plane_index, voxel_index = numpy.where(val_coords != -999)
    


    values  = val_coords[batch_index, plane_index, voxel_index]
    x_index = numpy.int32(x_coords[batch_index, plane_index, voxel_index])
    y_index = numpy.int32(y_coords[batch_index, plane_index, voxel_index])

    # print(numpy.min(x_index))
    # print(numpy.min(y_index))
    # print()
    # print(numpy.max(x_index))
    # print(numpy.max(y_index))

    # Fill in the output tensor

    # FOrmat should be [N, C, H, W]
    output_array[batch_index, plane_index, y_index, x_index] = values    

    return output_array

def larcvsparse_to_scnsparse_2d(input_array):
    # This format converts the larcv sparse format to
    # the tuple format required for sparseconvnet

    # First, we can split off the features (which is the pixel value)
    # and the indexes (which is everythin else)

    # To handle the multiplane networks, we have to split this into
    # n_planes and pass it out as a list

    n_planes = input_array.shape[1]
    batch_size = input_array.shape[0]



    raw_planes = numpy.split(input_array,n_planes, axis=1)

    output_list = []
    output_features = []
    output_dimension = []

    for i, plane in enumerate(raw_planes):
        # First, squeeze off the plane dimension from this image:
        plane = numpy.squeeze(plane, axis=1)

        # Next, figure out the x, y, value coordinates:
        y,x,features = numpy.split(plane, 3, axis=-1)

        # print("X: ",numpy.max(x))
        # print("Y: ", numpy.max(y))

        non_zero_locs = numpy.where(features != -999)

        # Pull together the different dimensions:
        x = x[non_zero_locs]
        y = y[non_zero_locs]
        p = numpy.full(x.shape, fill_value=i)
        features = features[non_zero_locs]
        features = numpy.expand_dims(features,axis=-1)

        batch = non_zero_locs[0]

        # dimension = numpy.concatenate([x,y,batch], axis=0)
        # dimension = numpy.stack([x,y,batch], axis=-1)
        dimension = numpy.stack([p,y,x,batch], axis=-1)

        output_features.append(features)
        output_dimension.append(dimension)

    output_features = numpy.concatenate(output_features)
    output_dimension = numpy.concatenate(output_dimension)


    output_list = [output_dimension, output_features, batch_size]

    return output_list


def larcvsparse_to_scnsparse_3d(input_array):
    # This format converts the larcv sparse format to
    # the tuple format required for sparseconvnet

    # First, we can split off the features (which is the pixel value)
    # and the indexes (which is everythin else)

    n_dims = input_array.shape[-1]

    split_tensors = numpy.split(input_array, n_dims, axis=-1)


    # To map out the non_zero locations now is easy:
    non_zero_inds = numpy.where(split_tensors[-1] != -999)

    # The batch dimension is just the first piece of the non-zero indexes:
    batch_size  = input_array.shape[0]
    batch_index = non_zero_inds[0]

    # Getting the voxel values (features) is also straightforward:
    features = numpy.expand_dims(split_tensors[-1][non_zero_inds],axis=-1)

    # Lastly, we need to stack up the coordinates, which we do here:
    dimension_list = []
    for i in range(len(split_tensors) - 1):
        dimension_list.append(split_tensors[i][non_zero_inds])

    # Tack on the batch index to this list for stacking:
    dimension_list.append(batch_index)

    # And stack this into one numpy array:
    dimension = numpy.stack(dimension_list, axis=-1)

    output_array = (dimension, features, batch_size,)
    return output_array


def larcvsparse_to_dense_3d(input_array, dense_shape):


    batch_size = input_array.shape[0]
    output_array = numpy.zeros((batch_size,1) + dense_shape, dtype=numpy.float32)

    # By default, this returns channels_first format with just one channel.
    # You can just reshape since it's an empty dimension, effectively

    x_coords   = input_array[:,:,0]
    y_coords   = input_array[:,:,1]
    z_coords   = input_array[:,:,2]
    val_coords = input_array[:,:,3]


    # Find the non_zero indexes of the input:
    batch_index, voxel_index = numpy.where(val_coords != -999)

    values  = val_coords[batch_index, voxel_index]
    x_index = numpy.int32(x_coords[batch_index, voxel_index])
    y_index = numpy.int32(y_coords[batch_index, voxel_index])
    z_index = numpy.int32(z_coords[batch_index, voxel_index])


    # Fill in the output tensor

    output_array[batch_index, 0, x_index, y_index, z_index] = values    

    return output_array





