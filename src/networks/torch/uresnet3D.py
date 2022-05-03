import torch
import torch.nn as nn

'''UResNet is implemented recursively here.

On the downsample pass, each layer receives a tensor as input.
There are a series of convolutional blocks (either conv + BN + Relu or residual blocks)
and then a downsampling step.

After the downsampling step, the output goes into the next lowest layer.  The next lowest
layer performs it's steps (recursively down the network) and then returns an upsampled
image.  So, each layer returns an image of the same resolution as it's input.

On the upsampling pass, the layer receives a downsampled image.  It performs a series of convolutions,
and merges across the downsampled layers either before or after the convolutions.

It then performs an upsampling step, and returns the upsampled tensor.

'''
from src.config.network import Connection, GrowthRate, DownSampling, UpSampling, ConvMode, Norm

class Block3D(nn.Module):

    def __init__(self, *, 
            inplanes, 
            outplanes, 
            kernel = [3,3], 
            padding=[1,1], 
            n_planes=1, 
            activation = nn.functional.leaky_relu,
            params):
        nn.Module.__init__(self)

        # print("Receive outplanes ", outplanes)

        if n_planes == 3:
            stride = [1, 1, 1]
            kernel = [3,] + kernel
            padding = [1,] + padding
        else:
            stride = [1, 1, 1]
            kernel = [1,] + kernel
            padding = [0,] + padding

        # padding = [0,1,1] if n_planes == 1 else [1,1,1]
        self.outplanes = outplanes
        self.conv = nn.Conv3d(
            in_channels  = inplanes,
            out_channels = outplanes,
            kernel_size  = kernel,
            stride       = stride,
            padding      = padding,
            bias         = params.bias)


        if params.normalization == Norm.batch:
            self._do_normalization = True
            self.norm = nn.BatchNorm3d(outplanes)
        elif params.normalization == Norm.layer:
            self._do_normalization = True
            self.norm = nn.LayerNorm(outplanes)
        else:
            self._do_normalization = False


        self.activation = torch.nn.Leaky_Relu(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        if self.do_batch_norm:
            out = self.bn(out)
        out = self.relu(out)
        return out



class ResidualBlock3D(nn.Module):

    def __init__(self, *, inplanes, outplanes, n_planes=1, kernel=[3,3], padding=[1,1], params):
        nn.Module.__init__(self)



        self.convolution_1 = Block(
            inplanes    = inplanes,
            outplanes   = outplanes,
            kernel      = kernel,
            padding     = padding,
            params      = params)

        self.convolution_2 = Block(
            inplanes    = inplanes,
            outplanes   = outplanes,
            kernel      = kernel,
            padding     = padding,
            activation  = torch.nn.Identity(),
            params      = params)




    def forward(self, x):
        residual = x

        out = self.convolution_1(x)

        out = self.convolution_1(out)


        out += residual
        out = self.leaky_relu(out)

        return out

class ResidualBlock3D(nn.Module):

    def __init__(self, *, inplanes, outplanes, n_planes=1, kernel = [3,3], padding=[1,1], params):
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
            bias         = params.bias)

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
            bias         = params.bias)

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
            bias         = params.bias)

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
            bias         = params.bias)

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


    def __init__(self, *, inplanes, n_blocks, n_planes, kernel, padding, params):
        torch.nn.Module.__init__(self)

        if not params.residual:
            self.blocks = [ Block3D(inplanes = inplanes, outplanes = inplanes,
                n_planes = n_planes, kernel=kernel, padding=padding, params = params) for i in range(n_blocks) ]
        else:
            self.blocks = [ ResidualBlock3D(inplanes = inplanes, outplanes = inplanes,
                n_planes = n_planes, kernel=kernel, padding=padding, params = params) for i in range(n_blocks)]

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


        self.merger = Block3D(
            outplanes = params.bottleneck_deepest,
            inplanes  = inplanes,
            kernel    = [1,1],
            padding   = [0,0],
            n_planes  = 3,
            params    = params
        )

        kernel  = [params.filter_size_deepest, params.filter_size_deepest]
        padding = [ int((k - 1) / 2) for k in kernel ]

        self.blocks = BlockSeries3D(
            inplanes = params.bottleneck_deepest,
            n_blocks = params.blocks_deepest_layer,
            n_planes = 1,
            kernel   = kernel,
            padding  = padding,
            params   = params,
        )



        self.splitter = Block3D(
            inplanes  = params.bottleneck_deepest,
            outplanes = inplanes,
            kernel    = [1,1],
            padding   = [0,0],
            n_planes  = 3,
            params    = params
        )

    def forward(self, x):
        x = self.merger(x)
        x = self.blocks(x)
        x = self.splitter(x)
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


        self.up = torch.nn.Upsample(scale_factor=(1,2,2), mode="nearest")

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
                                             kernel   = [3,3],
                                             padding  = [1,1],
                                             params   = params)

            if params.growth_rate == GrowthRate.multiplicative:
                n_filters_next = 2 * inplanes
            else:
                n_filters_next = inplanes + params.n_initial_filters


            # Down sampling operation:
            # This does change the number of filters from above down-pass blocks
            if params.downsampling == DownSampling.convolutional:
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
            if params.upsampling == UpSampling.convolutional:
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
                                           kernel   = [3,3],
                                           padding  = [1,1],
                                           params   = params)

            # Residual connection operation:
            if params.connections == Connection.sum:
                self.connection = SumConnection3D()
            elif params.connections == Connection.concat:
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

    def __init__(self, params, spatial_size):


        torch.nn.Module.__init__(self)



        torch.nn.Module.__init__(self)




        self.initial_convolution = Block3D(
            inplanes    = 1,
            outplanes   = params.n_initial_filters,
            kernel      = [5,5],
            padding     = [2,2],
            params      = params)


        n_filters = params.n_initial_filters
        # Next, build out the convolution steps:

        self.net_core = UNetCore3D(
            depth    = params.depth,
            inplanes = params.n_initial_filters,
            params   = params)

        # We need final output shaping too.
        # Even with shared weights, keep this separate:

        self.final_layer = BlockSeries3D(
            inplanes = params.n_initial_filters,
            n_blocks = params.blocks_final,
            n_planes = 1,
            kernel   = [3,3],
            padding  = [1,1],
            params   = params )


        self.bottleneck = nn.Conv3d(
            in_channels  = params.n_initial_filters,
            out_channels = 3,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            bias         = params.bias)

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
