import torch
import torch.nn as nn

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

class Block3D(nn.Module):

    def __init__(self, *, inplanes, outplanes, kernel = [3,3], padding=[1,1], n_planes=1, params):
        nn.Module.__init__(self)
        


        if n_planes == 3:
            stride = [1, 1, 1]
            kernel = [3,] + kernel
            padding = [1,1,1]
        else:
            stride = [1, 1, 1]
            kernel = [1,] + kernel
            padding = [0,] + padding

        # padding = [0,1,1] if n_planes == 1 else [1,1,1]

        self.conv = nn.Conv3d(
            in_channels  = inplanes,
            out_channels = outplanes,
            kernel_size  = kernel,
            stride       = stride,
            padding      = padding,
            bias         = params.use_bias)

        self.do_batch_norm = params.batch_norm
        
        if params.batch_norm:
            self.bn   = nn.BatchNorm3d(outplanes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        if self.do_batch_norm:
            out = self.bn(out)
        out = self.relu(out)
        return out


class ResidualBlock3D(nn.Module):

    def __init__(self, *, inplanes, outplanes, nplanes=1, params):
        nn.Module.__init__(self)


        if n_planes == 3:
            stride = [1, 1, 1]
            kernel = [3,] + kernel
            padding = [1,1,1]
        else:
            stride = [1, 1, 1]
            kernel = [1,] + kernel
            padding = [0,] + padding

        self.conv1 = nn.Conv3d(
            in_channels  = inplanes,
            out_channels = outplanes,
            kernel_size  = kernel,
            stride       = stride,
            padding      = padding,
            bias         = params.use_bias)

        self.do_batch_norm = params.batch_norm
        if self.do_batch_norm:
            self.bn1 = nn.BatchNorm3d(outplanes)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            in_channels  = outplanes,
            out_channels = outplanes,
            kernel_size  = kernel,
            stride       = stride,
            padding      = padding,
            bias         = params.use_bias)

        if self.do_batch_norm:
            self.bn2 = nn.BatchNorm3d(outplanes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.do_batch_norm:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        if self.do_batch_norm:
            out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class ConvolutionDownsample3D(nn.Module):

    def __init__(self, *, inplanes, outplanes, params):
        nn.Module.__init__(self)

        self.conv = nn.Conv3d(
            in_channels  = inplanes,
            out_channels = outplanes,
            kernel_size  = [1, 2, 2],
            stride       = [1, 2, 2],
            padding      = [0, 0, 0],
            bias         = params.use_bias)

        self.do_batch_norm = params.batch_norm
        if self.do_batch_norm:
            self.bn   = nn.BatchNorm3d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        if self.do_batch_norm:
            out = self.bn(out)
        out = self.relu(out)
        return out


class ConvolutionUpsample3D(nn.Module):

    def __init__(self, *, inplanes, outplanes, params):
        nn.Module.__init__(self)

        self.conv = nn.ConvTranspose3d(
            in_channels  = inplanes,
            out_channels = outplanes,
            kernel_size  = [1, 2, 2],
            stride       = [1, 2, 2],
            padding      = [0, 0, 0],
            bias         = params.use_bias)

        self.do_batch_norm = params.batch_norm
        if self.do_batch_norm:
            self.bn   = nn.BatchNorm3d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.conv(x)
        if self.do_batch_norm:
            out = self.bn(out)
        out = self.relu(out)
        return out


class BlockSeries3D(torch.nn.Module):


    def __init__(self, *, inplanes, n_blocks, n_planes, params):
        torch.nn.Module.__init__(self)

        if not params.residual:
            self.blocks = [ Block3D(inplanes = inplanes, outplanes = inplanes, 
                n_planes = n_planes, params = params) for i in range(n_blocks) ]
        else:
            self.blocks = [ ResidualBlock3D(inplanes = inplanes, outplanes = inplanes, 
                n_planes = n_planes, params = params) for i in range(n_blocks)]

        for i, block in enumerate(self.blocks):
            self.add_module('block_{}'.format(i), block)


    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

        return x


class DeepestBlock3D(nn.Module):

    def __init__(self, inplanes, params):
        nn.Module.__init__(self)


        # The deepest block concats across planes, applies convolutions,
        # Then splits into planes again

        # self.merger = nn.Conv3d(
        #     in_channels     = inplanes,
        #     out_channels    = FLAGS.NPLANES*inplanes,
        #     kernel_size     = [FLAGS.NPLANES,1,1],
        #     stride          = [1,1,1],
        #     padding         = [0, 0, 0],
        #     bias            = False)


        self.blocks = BlockSeries3D( 
            inplanes = inplanes, 
            n_blocks = params.blocks_deepest_layer, 
            n_planes = 3,
            params   = params,
        )

        # self.splitter = nn.ConvTranspose3d(
        #     in_channels     = FLAGS.NPLANES*inplanes,
        #     out_channels    = inplanes,
        #     kernel_size     = [FLAGS.NPLANES,1,1],
        #     stride          = [1,1,1],
        #     padding         = [0, 0, 0],
        #     bias            = False)




    def forward(self, x):
        # x = self.merger(x)
        x = self.blocks(x)
        # x = self.splitter(x)


        return x

class NoConnection3D(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x, residual):
        return x

class SumConnection3D(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x, residual):
        return x + residual

class ConcatConnection3D(nn.Module):

    def __init__(self, *, inplanes, params):
        nn.Module.__init__(self)

        self.bottleneck = Block3D(
            inplanes    = 2*inplanes,
            outplanes   = inplanes,
            kernel      = [1,1],
            padding     = [0,0],
            params      = params)

    def forward(self, x, residual):
        x = torch.cat([x, residual], dim=1)
        x = self.bottleneck(x)
        return x




class MaxPooling3D(nn.Module):

    def __init__(self,*, inplanes, outplanes, params):
        nn.Module.__init__(self)


        self.pool = torch.nn.MaxPool3d(stride=[1,2,2], kernel_size=[1,2,2])

        self.bottleneck = Block3D(
            inplanes    = inplanes,
            outplanes   = outplanes,
            kernel      = [1,1],
            padding     = [0,0],
            params      = params)

    def forward(self, x):
        x = self.pool(x)

        return self.bottleneck(x)

class InterpolationUpsample3D(nn.Module):

    def __init__(self, *, inplanes, outplanes, params):
        nn.Module.__init__(self)


        self.up = torch.nn.Upsample(scale_factor=[1,2,2], mode="bilinear")

        self.bottleneck = Block3D(
            inplanes    = inplanes,
            outplanes   = outplanes,
            kernel      = [1,1],
            padding     = [0,0],
            params      = params)

    def forward(self, x):
        x = self.up(x)
        return self.bottleneck(x)



class UNetCore3D(nn.Module):

    def __init__(self, * , depth, inplanes, params):
        nn.Module.__init__(self)


        self.layers = params.blocks_per_layer
        self.depth  = depth

        if depth == 0:
            self.main_module = DeepestBlock3D(inplanes = inplanes, 
                                              params   = params)
        else:
            # Residual or convolutional blocks, applied in series:
            self.down_blocks = BlockSeries3D(inplanes = inplanes, 
                                             n_blocks = self.layers, 
                                             n_planes = 1,
                                             params   = params)

            if params.growth_rate == "multiplicative":
                n_filters_next = 2 * inplanes
            else:
                n_filters_next = inplanes + params.n_initial_filters
                

            # Down sampling operation:
            # This does change the number of filters from above down-pass blocks
            if params.downsampling == "convolutional":
                self.downsample = ConvolutionDownsample3D(inplanes    = inplanes,
                                                          outplanes   = n_filters_next,
                                                          params      = params)
            else:
                self.downsample = MaxPooling3D(inplanes  = inplanes,
                                               outplanes = n_filters_next,
                                               params    = params)

            
            # Submodule:
            self.main_module    = UNetCore3D(depth    = depth-1, 
                                             inplanes = n_filters_next,
                                             params   = params)
            


            # Upsampling operation:
            if params.upsampling == "convolutional":
                self.upsample       = ConvolutionUpsample3D(inplanes  = n_filters_next,
                                                            outplanes = inplanes,
                                                            params    = params)
            else:
                self.upsample = InterpolationUpsample3D(inplanes  = n_filters_next,
                                                        outplanes = inplanes,
                                                        params    = params)

            # Convolutional or residual blocks for the upsampling pass:
            self.up_blocks = BlockSeries3D(inplanes = inplanes,
                                           n_blocks = self.layers,
                                           n_planes = 1,
                                           params   = params)

            # Residual connection operation:
            if params.connections == "sum":
                self.connection = SumConnection3()
            elif params.connections == "concat":
                self.connection = ConcatConnection3D(inplanes=inplanes, params=params)
            else:
                self.connection = NoConnection3D()


    def forward(self, x):

        # Take the input and apply the downward pass convolutions.  Save the residual
        # at the correct time.
        if self.depth != 0:


            x = self.down_blocks(x)

            residual = x

            # perform the downsampling operation:
            x = self.downsample(x)

        # Apply the main module:
        x = self.main_module(x)

        if self.depth != 0:

            # perform the upsampling step:
            # perform the downsampling operation:
            x = self.upsample(x)

            # Apply the convolutional steps:
            x = self.up_blocks(x)

            # Connect with the residual if necessary:
            x = self.connection(x, residual=residual)

        return x

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


class UResNet3D(torch.nn.Module):

    def __init__(self, * ,
                n_initial_filters,    # Number of initial filters in the network.
                batch_norm,           # Use Batch norm?
                use_bias,             # Use Bias layers?
                residual,             # Use residual blocks where possible
                depth,                # How many times to downsample and upsample
                blocks_final,         # How many blocks just before bottleneck?
                blocks_per_layer,     # How many blocks to apply at this layer, if not deepest
                blocks_deepest_layer, # How many blocks at the deepest layer
                connections,          # What type of connection?
                upsampling,           # What type of upsampling?
                downsampling,         # What type of downsampling?
                shape,                # Data shape
                bottleneck_deepest,   # How many filters to use in combined, deepest convolutions
                filter_size_deepest,  # What size filter to use in the deepest convolutions
                growth_rate,          # Either multiplicative (doubles) or additive (constant addition))
            ):


        torch.nn.Module.__init__(self)


        params = objectview({
            'n_initial_filters'     : n_initial_filters,
            'batch_norm'            : batch_norm,
            'use_bias'              : use_bias,
            'residual'              : residual,
            'depth'                 : depth,
            'blocks_final'          : blocks_final,
            'blocks_per_layer'      : blocks_per_layer,
            'blocks_deepest_layer'  : blocks_deepest_layer,
            'connections'           : connections,
            'upsampling'            : upsampling,
            'downsampling'          : downsampling,
            'shape'                 : shape,
            'growth_rate'           : growth_rate,
            'bottleneck_deepest'    : bottleneck_deepest,
            'filter_size_deepest'   : filter_size_deepest,
            })



        torch.nn.Module.__init__(self)




        self.initial_convolution = Block3D(
            inplanes    = 1,
            outplanes   = n_initial_filters,
            kernel      = [7,7],
            padding     = [3,3],
            params      = params)


        n_filters = n_initial_filters
        # Next, build out the convolution steps:

        self.net_core = UNetCore3D(
            depth    = depth,
            inplanes = n_initial_filters,
            params   = params)

        # We need final output shaping too.
        # Even with shared weights, keep this separate:

        self.final_layer = BlockSeries3D(
            inplanes = n_initial_filters,
            n_blocks = blocks_final,
            n_planes = 1,
            params   = params )


        self.bottleneck = nn.Conv3d(
            in_channels  = n_initial_filters,
            out_channels = 3,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            bias         = use_bias)

        # The rest of the final operations (reshape, softmax) are computed in the forward pass


        # # Configure initialization:
        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm3d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, input_tensor):


        batch_size = input_tensor.shape[0]



        # Reshape this tensor into the right shape to apply this multiplane network.

        x = input_tensor.view([batch_size, 1, input_tensor.shape[-3], input_tensor.shape[-2], input_tensor.shape[-1]])

        # Apply the initial convolutions:
        x = self.initial_convolution(x)

        # Apply the main unet architecture:
        x = self.net_core(x)

        # Apply the final residual block to each plane:
        x = self.final_layer(x)
        x = self.bottleneck(x)

        # To be compatible with the loss functions, all computed in 2D, we split here:

        # Break the images into 3 planes:
        x = torch.chunk(x, chunks=3, dim=2)
        x = [ _x.view(_x.shape[0], _x.shape[1], _x.shape[-2], _x.shape[-1]) for _x in x ]


        return x
