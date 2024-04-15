import jax
from jax import numpy
import flax.linen as nn


'''UResNet is implemented recursively here.

On the downsample pass, each layer receives a tensor as input.
There are a series of convolutional blocks (either conv + BN + Relu or residual blocks)
and then a downsampling step.

After the downsampling step, the output goes into the next lowest layer.  The next lowest
layer performs it's steps (recursively down the network) and then returns an upsampled
image.  So, each layer returns an image of the same resolution as it's input.

On the upsampling pass, the layer recieves a downsampled image.  It performs a series of convolutions,
and merges across the downsampled layers either before or after the convolutions.

It then performs an upsampling step, and returns the upsampled tensor.

'''

from src.config.network import Connection, GrowthRate, BlockStyle
from src.config.network import DownSampling, UpSampling, Norm

from functools import partial

from typing import Tuple, Any

class Block(nn.Module):

    outplanes:     int 
    strides:       Tuple[int]
    padding:       Tuple[int]
    kernel:        Tuple[int]
    activation:    callable
    normalization: Norm
    groups:        int
    bias:          bool

    @nn.compact
    def __call__(self, x, training: bool ):
        
        # Apply the convolution:
        x = nn.Conv(
            features     = self.outplanes,
            kernel_size  = self.kernel,
            strides      = self.strides,
            padding      = self.padding,
            feature_group_count = self.groups,
            use_bias     = self.bias)(x)
        

        # Apply the normalization:
        if self.normalization == Norm.batch:
            raise Exception("Batch Norm not implemented in JAX")
            # x = nn.BatchNorm(use_running_average= not training)(x)
        elif self.normalization == Norm.group:
            x = nn.GroupNorm(num_groups=4)(x)
        elif self.normalization == Norm.layer:
            x = nn.GroupNorm(num_groups=1)(x)
        elif self.normalization == Norm.instance:
            # print(x)
            # print(x.mean())
            x = nn.GroupNorm(group_size=1, num_groups=None)(x)

        return self.activation(x)


class ConvolutionUpsample(nn.Module):

    outplanes:     int 
    strides:       Tuple[int]
    padding:       Tuple[int]
    kernel:        Tuple[int]
    activation:    callable
    normalization: Norm
    groups:        int

    @nn.compact
    def __call__(self, x, training: bool = True):

        self.conv = nn.ConvTranspose(
            features     = self.outplanes,
            kernel_size  = self.kernel,
            stride       = self.strides,
            padding      = self.padding,
            feature_group_count = self.groups,
            use_bias     = self.bias)(x)

        # Apply the normalization:
        if self.normalization == Norm.batch:
            raise Exception("Batch Norm not implemented in JAX")
            # x = nn.BatchNorm(use_running_average= not training)(x)
        elif self.normalization == Norm.group:
            x = nn.GroupNorm(num_groups=4)(x)
        elif self.normalization == Norm.layer:
            x = nn.GroupNorm(num_groups=1)(x)
        elif self.normalization == Norm.instance:
            x = nn.GroupNorm(group_size=1)(x)

        return self.activation(x)



class ResidualBlock(nn.Module):

    outplanes:     int 
    strides:       Tuple[int]
    padding:       Tuple[int]
    kernel:        Tuple[int]
    activation:    callable
    normalization: Norm
    groups:        int
    bias:          bool

    @nn.compact
    def __call__(self, x, training: bool = True):
        
        inputs = x
        x = Block(
            outplanes     = self.outplanes,
            strides       = self.strides,
            padding       = self.padding,
            kernel        = self.kernel,
            activation    = self.activation,
            normalization = self.normalization,
            groups        = self.groups,
            bias          = self.bias,
        )(x, training = training)

        x = Block(
            outplanes     = self.outplanes,
            strides       = self.strides,
            padding       = self.padding,
            kernel        = self.kernel,
            activation    = lambda x : x,
            normalization = self.normalization,
            groups        = self.groups,
            bias          = self.bias,
        )(x, training = training)

        return self.activation(inputs + x)



# class ConvNextBlock(nn.Module):

#     outplanes:     int 
#     strides:       Tuple[int]
#     padding:       Tuple[int]
#     kernel:        Tuple[int]
#     activation:    callable
#     override_norm: bool
#     normalization: Norm
#     groups:        int

#     @nn.compact
#     def __call__(self, x, training : bool = True):

#         x = Block(
#             outplanes     = self.outplanes,
#             strides       = self.strides,
#             padding       = self.padding,
#             kernel        = self.kernel,
#             activation    = lambda x : x,
#             override_norm = self.override_norm,
#             normalization = self.normalization,
#             groups        = self.groups,
#         )(x, training = training)

#         x = Block(
#             outplanes     = self.outplanes,
#             strides       = self.strides,
#             padding       = self.padding,
#             kernel        = self.kernel,
#             activation    = lambda x : x,
#             override_norm = self.override_norm,
#             normalization = self.normalization,
#             groups        = self.groups,
#         )(x, training = training)


#     def __init__(self, *, inplanes, outplanes, params):

#         nn.Module.__init__(self)


#         kernel_1  = [params.kernel_size,params.kernel_size]
#         padding_1 = tuple( int((k - 1) / 2) for k in kernel_1 )

#         if params.depthwise:
#             groups = inplanes
#         else:
#             groups = 1

#         self.convolution_1 = Block(
#             inplanes    = inplanes,
#             outplanes   = inplanes,
#             kernel      = kernel_1,
#             padding     = padding_1,
#             groups      = groups,
#             activation  = torch.nn.Identity(),
#             params      = params)


#         kernel_2  = [1, 1]
#         padding_2 = [0, 0]



#         self.convolution_2 = Block(
#             inplanes    = inplanes,
#             outplanes   = 4 * inplanes,
#             kernel      = kernel_2,
#             padding     = padding_2,
#             override_norm = "none", # Sets to no normalization, as opposed to the default param
#             params      = params)

#         self.convolution_3 = Block(
#             inplanes    = 4 * inplanes,
#             outplanes   = outplanes,
#             kernel      = kernel_2,
#             padding     = padding_2,
#             activation  = torch.nn.Identity(),
#             override_norm = "", # Sets to no normalization, as opposed tothe default param
#             params      = params)


#     def forward(self, x):

#         residual = x

#         out = self.convolution_1(x)

#         out = self.convolution_2(out)

#         out = self.convolution_3(out)

#         out += residual

#         return out
    

class BlockSeries(nn.Module):

    n_blocks: int
    params: Any
    activation: callable

    @nn.compact
    def __call__(self, x, training: bool = True):
        '''
        x is expected to be an Array, not a list!
        '''
        inplanes = x[0].shape[-1]
        kernel = [self.params.kernel_size, self.params.kernel_size]


        if self.params.block_style == BlockStyle.none:
            for i in range(self.n_blocks):
                x = Block(
                    outplanes    = inplanes,
                    strides      = [1,1],
                    padding      = "SAME",
                    kernel       = kernel,
                    activation   = self.activation,
                    normalization= self.params.normalization,
                    groups       = inplanes if self.params.depthwise else 1,
                    bias         = self.params.bias,
                )(x, training)

        elif self.params.block_style == BlockStyle.residual:
            for i in range(self.n_blocks):
                x = ResidualBlock(
                    outplanes    = inplanes,
                    strides      = [1,1],
                    padding      = "SAME",
                    kernel       = kernel,
                    activation   = self.activation,
                    normalization= self.params.normalization,
                    groups       = inplanes if self.params.depthwise else 1,
                    bias         = self.params.bias,
                )(x, training)
                

        elif self.params.block_style == BlockStyle.convnext:
            raise Exception("Not yet implemented")


        return x

    

class DeepestBlock(nn.Module):

    params: Any
    activation: callable

    @nn.compact
    def __call__(self, x, training: bool = True):

        if self.params.block_concat:
            raise Exception("Not implemented in JAX")

        # Concatenate the inputs:
        x = numpy.concatenate(x, axis=-1)
        
        # Save the number of in-planes:
        inplanes = x.shape[-1]

        # Bottle neck:
        x = Block(
            outplanes    = self.params.bottleneck_deepest,
            strides      = [1,1],
            padding      = "SAME",
            kernel       = (1,1),
            activation   = self.activation,
            normalization= self.params.normalization,
            groups       = 1,
            bias         = self.params.bias,
        )(x, training)

        # Blocks:
        x = BlockSeries(
            n_blocks   = self.params.blocks_deepest_layer,
            params     = self.params,
            activation = self.activation, 
        )(x, training)

        # Un-bottleneck:
        x = Block(
            outplanes    = inplanes,
            strides      = [1,1],
            padding      = "SAME",
            kernel       = (1,1),
            activation   = self.activation,
            normalization= self.params.normalization,
            groups       = 1,
            bias         = self.params.bias,
        )(x, training)

        # This is where the classification head peels off:
        classification_head = x

        x = numpy.split(x, indices_or_sections=3, axis=-1)

        return x, classification_head, None # The none is a placeholder for vertex ID YOLO



class NoConnection(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x, residual):
        return x

class SumConnection(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x, residual):
        return x + residual

class ConcatConnection(nn.Module):


    params: Any
    outplanes: int
    activation: callable

    @nn.compact
    def __call__(self, x, res, training : bool = True):

        x = numpy.concatenate([x, res], axis=-1)

        x = Block(
            outplanes    = self.outplanes,
            strides      = [1,1],
            padding      = "SAME",
            kernel       = (1,1),
            activation   = self.activation,
            normalization= self.params.normalization,
            groups       = 1,
            bias         = self.params.bias,
        )(x, training)

        return x
    



class MaxPooling(nn.Module):
    params: Any
    outplanes: int
    activation: callable

    @nn.compact
    def __call__(self, x, training:bool = True):

        x = nn.max_pool(x, window_shape=(2,2), strides=(2,2))

        inplanes = x.shape[-1]

        x = Block(
            outplanes    = self.outplanes,
            strides      = [1,1],
            padding      = "SAME",
            kernel       = (1,1),
            activation   = self.activation,
            normalization= self.params.normalization,
            groups       = 1,
            bias         = self.params.bias,
        )(x, training)

        return x
    
class InterpolationUpsample(nn.Module):

    outplanes: int
    params: Any
    activation: callable

    @nn.compact
    def __call__(self, x, training : bool = True):

        input_size = x.shape[1:3]
        # CHANGED HERE TO AVOID VMAP
        output_size = [x.shape[0]] + [ 2 * i for i in input_size ] + [x.shape[-1]]
        # print("Target shape: ", output_size)
        # print("Pre shape: ", x.shape)
        x = jax.image.resize(x, output_size, method="bilinear")
        # print("Post shape: ", x.shape)

        x = Block(
            outplanes    = self.outplanes,
            strides      = [1,1],
            padding      = "SAME",
            kernel       = (1,1),
            activation   = self.activation,
            normalization= self.params.normalization,
            groups       = 1,
            bias         = self.params.bias,
        )(x, training)

        return x


class UNetCore(nn.Module):

    params: Any
    activation: callable
    depth: int

    @nn.compact
    def __call__(self, x, training: bool = True):


        # Set up the main module before beginning:
        if self.depth == 0:
            main_module = DeepestBlock(
                self.params, 
                self.activation
            )
        else:
            main_module = UNetCore(
                params = self.params,
                activation = self.activation,
                depth = self.depth - 1,
            )

        if self.depth != 0:
            # Store the residual:
            residual = x

            down_blocks = BlockSeries(
                n_blocks   = self.params.blocks_per_layer,
                params     = self.params,
                activation = self.activation,   
            )
            x = [ down_blocks(_x, training) for _x in x ]
        
            # Downsample:
            current_filters = x[0].shape[-1]
            if self.params.growth_rate == GrowthRate.multiplicative:
                n_filters_next = 2 *current_filters
            else:
                # Add to the number of filters
                n_filters_next =current_filters + self.params.n_initial_filters

            

            if self.params.downsampling == DownSampling.convolutional:
                downsample = Block(
                    outplanes    = n_filters_next,
                    strides      = [2,2],
                    padding      = [0,0],
                    kernel       = [2,2],
                    activation   = self.activation,
                    normalization= self.params.normalization,
                    groups       = 1
                )
            else:
                downsample = MaxPooling(self.params, n_filters_next, self.activation)
            x = [ downsample(_x, training) for _x in x]

        # Apply the main module:
        x, classification_head, vertex_head = main_module(x)
        
        if self.depth != 0:

            # Upsample
            if self.params.upsampling == UpSampling.convolutional:
                upsample = ConvolutionUpsample(
                    outplanes  = current_filters,
                    params     = self.params,
                    activation = self.activation,        
                )
            else:
                upsample = InterpolationUpsample(
                    outplanes  = current_filters,
                    params     = self.params,
                    activation = self.activation,    
                )
                
            x = [ upsample(_x, training) for _x in x]

            # Connection
            if self.params.connections == Connection.sum:
                connection = lambda x, r, training: x + r
            elif self.params.connections == Connection.concat:
                connection = ConcatConnection(
                    outplanes  = current_filters,
                    params     = self.params,
                    activation = self.activation,    
                )
            else:
                connection = lambda x, r, training: x

            x = [ connection(_x, _r, training) for _x, _r in zip(x, residual) ]

            # Up Blocks:
            up_block = BlockSeries(
                n_blocks   = self.params.blocks_per_layer,
                params     = self.params,
                activation = self.activation,   
            )
            x = [ up_block(_x, training) for _x in x]

        # Read out the vertex head if it's the right layer:
        if self.depth == self.params.vertex.depth: vertex_head = x

        # Return:
        return x, classification_head, vertex_head




class UResNet(nn.Module):

    params: Any
    spatial_size: Tuple[int]
    activation: callable

    @nn.compact
    def __call__(self, x, training : bool = True):
        '''
        Compute the call of the network,
        implicitly going to vmap over the batch size!
        '''

        spatial_size = x.shape[0:2]

        return_dict = {
            "event_label" : None,
            "vertex"      : None,
        }


        # Reshape this tensor into the right shape to apply this multiplane network.

        # Flax expects channels last:
        x = numpy.split(x, indices_or_sections=3, axis=-1)


        # Apply the initial convolutions:
        initial_conv = Block(
            outplanes     = self.params.n_initial_filters,
            strides       = [1, 1],
            padding       = "SAME",
            kernel        = [5, 5],
            activation    = self.activation,
            normalization = self.params.normalization,
            groups        = 1,
            bias          = self.params.bias,
        )
        x = [ initial_conv(_x, training) for _x in x ]


        # Apply the main unet architecture:
        seg_labels, classification_head, vertex \
            = UNetCore(self.params, self.activation, self.params.depth)(x, training)


        final_layer = BlockSeries(
            n_blocks   = self.params.blocks_final,
            params     = self.params,
            activation = self.activation)


        bottleneck = nn.Conv(
            features     = 3,
            kernel_size  = 1,
            strides      = 1,
            padding      = 0,
            feature_group_count = 1,
            use_bias     = self.params.bias)
        


        # Apply the final residual block to each plane:
        seg_labels = [ final_layer(_x) for _x in seg_labels ]
        seg_labels = [ bottleneck(_x) for _x in seg_labels ]

        # Always return the segmentation
        return_dict["segmentation"] = seg_labels

        # The rest of the final operations (reshape, softmax) are computed in the forward pass
        if self.params.classification.active:

            if self.params.classification.detach:
                classification_head = jax.lax.stop_gradient(classification_head)

            classification_x = nn.Conv(
                features     = self.params.classification.n_filters,
                kernel_size  = 1,
                strides      = 1,
                padding      = 0,
                feature_group_count = 1,
                use_bias     = self.params.bias)(classification_head)

            classification_x = BlockSeries(
                n_blocks   = self.params.classification.n_layers,
                params     = self.params,
                activation = self.activation)(classification_x, training)

            classification_x = nn.Conv(
                features     = 4,
                kernel_size  = 1,
                strides      = 1,
                padding      = 0,
                feature_group_count = 1,
                use_bias     = self.params.bias)(classification_x)

            classification_x = nn.avg_pool(
                classification_x,
                classification_x.shape[0:2]
            )
            
            return_dict["event_label"] = classification_x.reshape((4,))

        if self.params.vertex.active:

            if self.params.vertex.detach:
                vertex = jax.lax.stop_gradient(vertex)

            vertex_input = nn.Conv(
                features     = self.params.vertex.n_filters,
                kernel_size  = 1,
                strides      = 1,
                padding      = 0,
                feature_group_count = 1,
                use_bias     = self.params.bias)

            vertex = [ vertex_input(v) for v in vertex ]


            vertex_layers = BlockSeries(
                n_blocks   = self.params.vertex.n_layers,
                params     = self.params,
                activation = self.activation)
            
            vertex = [ vertex_layers(v, training) for v in vertex ]


            bottleneck_vertex =  nn.Conv(
                features     = 3,
                kernel_size  = 1,
                strides      = 1,
                padding      = 0,
                feature_group_count = 1,
                use_bias     = self.params.bias)

            vertex = [ bottleneck_vertex(v) for v in vertex]

            # No sigmoid applied in JAX to make the loss easier to compute!
            # return_dict["vertex"] = [ jax.nn.sigmoid(v) for v in vertex ]
            return_dict["vertex"] = vertex


        return return_dict

