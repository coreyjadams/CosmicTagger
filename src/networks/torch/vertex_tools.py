import torch


def create_vertex_meta(args, image_meta, image_shape):

    # To predict the vertex, we first figure out the size of each bounding box:
    # Vertex comes out with shape :
    # [batch_size, channels, max_boxes, 2*ndim (so 4, in this case)]
    vertex_depth = args.network.depth - args.network.vertex.depth
    vertex_output_space = tuple(d // 2**vertex_depth  for d in image_shape )
    anchor_size = image_meta['size'] / vertex_output_space

    origin = image_meta['origin']

    return {
        "origin"              : origin,
        "vertex_output_space" : vertex_output_space,
        "anchor_size"         : anchor_size
    }


def predict_vertex(network_dict, vertex_meta):

    # We also flatten to make the argmax operation easier:
    detection_logits = [ n[:,0,:,:].reshape((n.shape[0], -1)) for n in  network_dict['vertex'] ]

    # Extract the index, which comes out flattened:
    predicted_vertex_index = [ torch.argmax(n, dim=1) for n in detection_logits ]


    # Convert flat index to 2D coordinates:
    height_index = [torch.div(p, vertex_meta['vertex_output_space'][1], rounding_mode='floor')  for p in predicted_vertex_index]
    width_index  = [p % vertex_meta['vertex_output_space'][1]  for p in predicted_vertex_index]

    # Extract the regression parameters for every box:
    internal_offsets_height = [ n[:,1,:,:].reshape((n.shape[0], -1)) for n in  network_dict['vertex'] ]
    internal_offsets_width  = [ n[:,2,:,:].reshape((n.shape[0], -1)) for n in  network_dict['vertex'] ]

    # Get the specific regression parameters
    # Creates flat index into the vectors
    batch_size = network_dict['vertex'][0].shape[0]
    batch_indexes = torch.arange(batch_size)

    # Extract the specific internal offsets:
    internal_offsets_height = [ i[batch_indexes, p] for i, p in zip(internal_offsets_height, predicted_vertex_index) ]
    internal_offsets_width  = [ i[batch_indexes, p] for i, p in zip(internal_offsets_width,  predicted_vertex_index) ]

    # Calculate the predicted height as origin + (p+r)*box_size
    predicted_height = [ vertex_meta['origin'][i,0] + (p+r)*vertex_meta['anchor_size'][i,0] for  \
        i, (p,r) in enumerate(zip(height_index, internal_offsets_height)) ]
    predicted_width  = [ vertex_meta['origin'][i,1] + (p+r)*vertex_meta['anchor_size'][i,1] for \
        i, (p,r) in enumerate(zip(width_index, internal_offsets_width)) ]

    # Stack it all together properly:
    vertex_prediction = torch.stack([
        torch.stack(predicted_height, dim=-1),
        torch.stack(predicted_width, dim=-1)
    ], dim=-1)

    return vertex_prediction
