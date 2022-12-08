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

def event_label(neutrino_particles, n_neutrino_pixels, neutrino_threshold=10):

    # Select electron neutrinos:
    nu_e  = numpy.abs(neutrino_particles['_pdg'] == 12)
    nu_mu = numpy.abs(neutrino_particles['_pdg'] == 14)

    # Neutral Current or Charged Current?
    nc = neutrino_particles["_current_type"] == 1

    n_events = neutrino_particles.shape[0]

    labels = numpy.zeros(n_events, dtype="int32")

    cosmic_labels = n_neutrino_pixels < neutrino_threshold

    labels[nu_e] = 0
    labels[nu_mu] = 1
    labels[nc] = 2
    labels[cosmic_labels] = 3

    return labels

            # minibatch_data['vertex2d'] = data_transforms.vertex_projection(
            #     minibatch_data['particle'][:,0]
            # )

            # minibatch_data['event_label'] = data_transforms.event_label(
            #     minibatch_data['particle'][:,0]
            # )


def larcvsparse_to_dense_2d(input_array, dense_shape, dataformat):

    batch_size = input_array.shape[0]
    n_planes   = input_array.shape[1]

    if dataformat == "channels_first":
        output_array = numpy.zeros((batch_size, n_planes, dense_shape[0], dense_shape[1]), dtype=numpy.float32)
    else:
        output_array = numpy.zeros((batch_size, dense_shape[0], dense_shape[1], n_planes), dtype=numpy.float32)


    x_coords = input_array[:,:,:,0]
    y_coords = input_array[:,:,:,1]
    val_coords = input_array[:,:,:,2]


    filled_locs = val_coords != -999
    non_zero_locs = val_coords != 0.0
    mask = numpy.logical_and(filled_locs,non_zero_locs)
    # Find the non_zero indexes of the input:
    batch_index, plane_index, voxel_index = numpy.where(filled_locs)


    values  = val_coords[batch_index, plane_index, voxel_index]
    x_index = numpy.int32(x_coords[batch_index, plane_index, voxel_index])
    y_index = numpy.int32(y_coords[batch_index, plane_index, voxel_index])

    # print(numpy.min(x_index))
    # print(numpy.min(y_index))
    # print()
    # print(numpy.max(x_index))
    # print(numpy.max(y_index))

    # Tensorflow expects format as either [batch, height, width, channel]
    # or [batch, channel, height, width]
    # Fill in the output tensor
    if dataformat == "channels_first":
        # output_array[batch_index, plane_index, y_index, x_index] = values
        output_array[batch_index, plane_index, y_index, x_index] = values
    else:
        output_array[batch_index, y_index, x_index, plane_index] = values


    return output_array

def larcvsparse_to_scnsparse_2d(input_array):
    # This format converts the larcv sparse format to
    # the tuple format required for sparseconvnet

    # First, we can split off the features (which is the pixel value)
    # and the indexes (which is everything else)

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
        x,y,features = numpy.split(plane, 3, axis=-1)

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


def form_yolo_targets(vertex_depth, vertex_labels, particle_labels, event_labels, dataformat, image_meta, ds):

    batch_size = event_labels.shape[0]
    event_energy = particle_labels['_energy_init'][:,0]

    # Vertex comes out with shape [batch_size, channels, max_boxes, 2*ndim (so 4, in this case)]
    image_shape = [ int(i /ds) for i in image_meta['full_pixels'][0] ]

    vertex_labels = vertex_labels[:,:,0,0:2]
    # The data gets loaded in (W, H) format and we need it in (H, W) format.:
    vertex_labels[:,:,[0,1]] = vertex_labels[:,:,[1,0]]


    # First, determine the dimensionality of the output space of the vertex yolo network:
    vertex_output_space = tuple(d // 2**vertex_depth  for d in image_shape )


    if dataformat == "channels_last":
        vertex_labels = numpy.transpose(vertex_channels_first,(0,2,1))
        vertex_presence_labels = numpy.zeros((batch_size,) + vertex_output_space + (3,), dtype="float32")
    else:
        # Nimages, 3 planes, shape-per-plane
        vertex_presence_labels = numpy.zeros((batch_size, 3,) + vertex_output_space, dtype="float32")


    n_pixels_vertex = 2**vertex_depth


    # To create the right bounding box location, we have to map the vertex x/z/y to a set of pixels.

    corrected_vertex_position = vertex_labels + image_meta["origin"]
    fractional_vertex_position = corrected_vertex_position / image_meta["size"]


    vertex_output_space_anchor_box_float = vertex_output_space * fractional_vertex_position

    vertex_output_space_anchor_box = vertex_output_space_anchor_box_float.astype("int")


    # This part creates indexes into the presence labels values:
    batch_index = numpy.arange(batch_size).repeat(3) # 3 for 3 planes
    plane_index = numpy.tile(numpy.arange(3), batch_size) # Tile 3 times for 3 planes

    h_index = numpy.concatenate(vertex_output_space_anchor_box[:,:,0])
    w_index = numpy.concatenate(vertex_output_space_anchor_box[:,:,1])



    if dataformat == "channels_last":
        vertex_presence_labels[batch_index, h_index, w_index, plane_index] = 1.0
    else:
        vertex_presence_labels[batch_index, plane_index, h_index, w_index] = 1.0



    # Finally, we should exclude any event that is labeled "cosmic only" from having a vertex
    # truth label:
    cosmics = event_labels == 3
    vertex_presence_labels[cosmics,:,:,:] = 0.0




    # Now, compute the location inside of an achor box for x/y.
    # Normalize to (0,1)

    bounding_box_location = vertex_output_space_anchor_box_float - vertex_output_space_anchor_box

    bounding_box_location = bounding_box_location.astype("float32")

    if dataformat == "channels_last":
        vertex_presence_labels = numpy.split(vertex_presence_labels, 3, axis=-1)
        vertex_presence_labels = [v.reshape((batch_size, ) + vertex_output_space) for v in vertex_presence_labels]
    else:
        vertex_presence_labels = numpy.split(vertex_presence_labels, 3, axis=1)
        vertex_presence_labels = [v.reshape((batch_size, ) + vertex_output_space) for v in vertex_presence_labels]


    return {
        "detection"  : vertex_presence_labels,
        "regression" : bounding_box_location,
        "energy"     : event_energy,
        "xy_loc"     : vertex_labels,
    }
