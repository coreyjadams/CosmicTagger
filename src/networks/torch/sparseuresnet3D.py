import torch
import torch.nn as nn
import sparseconvnet as scn

from src import utils

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
from src.config.network import Connection, GrowthRate, DownSampling, UpSampling, Norm

class SparseBlock(nn.Module):

    def __init__(self, *, inplanes, outplanes, nplanes=1, params):

        nn.Module.__init__(self)

        self.conv1 = scn.SubmanifoldConvolution(dimension=3,
            nIn=inplanes,
            nOut=outplanes,
            filter_size=[nplanes,3,3],
            bias=params.bias)

        if params.normalization == Norm.batch:
            self._do_normalization = True
            self.relu = scn.BatchNormLeakyReLU(outplanes)
        elif params.normalization == Norm.layer:
            raise Exception("Layer norm not supported in SCN")
        else:
            self.relu = scn.LeakyReLU()

    def forward(self, x):

        out = self.conv1(x)
        out = self.relu(out)

        return out



class SparseResidualBlock(nn.Module):

    def __init__(self, *, inplanes, outplanes, nplanes=1, params):
        nn.Module.__init__(self)


        self.conv1 = scn.SubmanifoldConvolution(dimension=3,
            nIn         = inplanes,
            nOut        = outplanes,
            filter_size = [nplanes,3,3],
            bias        = params.bias)

        if params.normalization == Norm.batch:
            self._do_normalization = True
            self.relu1 = scn.BatchNormLeakyReLU(outplanes)
        elif params.normalization == Norm.layer:
            raise Exception("Layer norm not supported in SCN")
        else:
            self.relu1 = scn.LeakyReLU()

        self.conv2 = scn.SubmanifoldConvolution(dimension=3,
            nIn         = outplanes,
            nOut        = outplanes,
            filter_size = [nplanes,3,3],
            bias        = False)

        if params.normalization == Norm.batch:
            self._do_normalization = True
            self.norm2 = scn.BatchNormalization(outplanes)


        self.residual = scn.Identity()
        self.relu2 = scn.LeakyReLU()

        self.add = scn.AddTable()

    def forward(self, x):

        residual = self.residual(x)

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)

        if self._do_normalization:
            out = self.norm2(out)

        # The addition of sparse tensors is not straightforward, since

        out = self.add([out, residual])

        out = self.relu2(out)

        return out




class SparseConvolutionDownsample(nn.Module):

    def __init__(self, *, inplanes, outplanes,nplanes=1, params):
        nn.Module.__init__(self)

        self.conv = scn.Convolution(dimension=3,
            nIn             = inplanes,
            nOut            = outplanes,
            filter_size     = [nplanes,2,2],
            filter_stride   = [1,2,2],
            bias            = params.bias
        )

        if params.normalization == Norm.batch:
            self.relu = scn.BatchNormLeakyReLU(outplanes)
        elif params.normalization == Norm.layer:
            raise Exception("Layer norm not supported in SCN")
        else:
            self.relu = scn.LeakyReLU()

    def forward(self, x):
        out = self.conv(x)

        out = self.relu(out)
        return out


class SparseConvolutionUpsample(nn.Module):

    def __init__(self, *, inplanes, outplanes, nplanes=1, params):
        nn.Module.__init__(self)

        self.conv = scn.Deconvolution(dimension=3,
            nIn             = inplanes,
            nOut            = outplanes,
            filter_size     = [nplanes,2,2],
            filter_stride   = [1,2,2],
            bias            = params.bias
        )

        if params.normalization == Norm.batch:
            self.relu = scn.BatchNormLeakyReLU(outplanes)
        elif params.normalization == Norm.layer:
            raise Exception("Layer norm not supported in SCN")
        else:
            self.relu = scn.LeakyReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out

class SparseBlockSeries(torch.nn.Module):


    def __init__(self, *, inplanes, n_blocks, n_planes=1, params):
        torch.nn.Module.__init__(self)

        if params.residual:
            self.blocks = [ SparseResidualBlock(inplanes = inplanes,
                                                outplanes = inplanes,
                                                nplanes = n_planes,
                                                params = params)
                                for i in range(n_blocks)
                            ]
        else:
            self.blocks = [ SparseBlock(inplanes = inplanes,
                                        outplanes = inplanes,
                                        nplanes = n_planes,
                                        params = params)
                                for i in range(n_blocks)]

        for i, block in enumerate(self.blocks):
            self.add_module('block_{}'.format(i), block)


    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x


class SparseDeepestBlock(nn.Module):

    def __init__(self, *, inplanes, params):
        nn.Module.__init__(self)


        # The deepest block applies convolutions that act on all three planes together

        # First we apply a convolution to map all three planes into 1 plane (of the same spatial size)

        self.merger = scn.Convolution(dimension=3,
            nIn             = inplanes,
            nOut            = params.bottleneck_deepest,
            filter_size     = [3,1,1],
            filter_stride   = [1,1,1],
            bias            = params.bias)


        self.blocks = SparseBlockSeries(inplanes = params.bottleneck_deepest,
                                        n_blocks =  params.blocks_deepest_layer,
                                        n_planes = 1,
                                        params   = params)

        self.splitter = scn.Deconvolution(dimension=3,
            nIn             = params.bottleneck_deepest,
            nOut            = inplanes,
            filter_size     = [3,1,1],
            filter_stride   = [1,1,1],
            bias            = params.bias)


    def forward(self, x):

        return x

        x = self.merger(x)
        x = self.blocks(x)
        x = self.splitter(x)
        return x


class NoConnection(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x, residual):
        return x

class SumConnection(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.op = scn.AddTable()

    def forward(self, x, residual):
        return self.op([x, residual])

class ConcatConnection(nn.Module):

    def __init__(self, *, inplanes, params):
        nn.Module.__init__(self)

        self.concat = scn.JoinTable()
        self.bottleneck = scn.SubmanifoldConvolution(3,
                            nIn         = 2*inplanes,
                            nOut        = inplanes,
                            filter_size = 1,
                            bias        = params.bias)

    def forward(self, x, residual):
        x = self.concat([x, residual])
        return self.bottleneck(x)


class SparseUNetCore(nn.Module):

    def __init__(self,  *,  depth, inplanes,  params):
        nn.Module.__init__(self)


        self.depth  = depth

        if depth == 0:
            self.main_module = SparseDeepestBlock(inplanes=inplanes, params=params)
        else:
            # Residual or convolutional blocks, applied in series:
            self.down_blocks = SparseBlockSeries(inplanes = inplanes,
                                                 n_blocks = params.blocks_per_layer,
                                                 params   = params)


            if params.growth_rate == "multiplicative":
                n_filters_next = 2 * inplanes
            else:
                n_filters_next = inplanes + params.n_initial_filters


            # Down sampling operation:
            self.downsample  = SparseConvolutionDownsample(inplanes  = inplanes,
                                                           outplanes = n_filters_next,
                                                           nplanes   = 1,
                                                           params    = params)

            # Submodule:
            self.main_module = SparseUNetCore(depth    = depth-1,
                                              inplanes = n_filters_next,
                                              params   = params)


            # Upsampling operation:
            self.upsample    = SparseConvolutionUpsample(inplanes = n_filters_next,
                                                         outplanes = inplanes,
                                                         params = params)


            # Convolutional or residual blocks for the upsampling pass:
            self.up_blocks = SparseBlockSeries(inplanes = inplanes,
                                               n_blocks = params.blocks_per_layer,
                                               params   = params)

            # Residual connection operation:
            if params.connections == Connection.sum:
                self.connection = SumConnection()
            elif params.connections == Connection.concat:
                self.connection = ConcatConnection(inplanes=inplanes, params=params)
            else:
                self.connection = NoConnection()


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

            # Connect with the residual if necessary:
            x = self.connection(x, residual)

            # Apply the convolutional steps:
            x = self.up_blocks(x)


        return x



class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


class UResNet3D(torch.nn.Module):

    def __init__(self, params, spatial_size):
        torch.nn.Module.__init__(self)

        # Create the sparse input tensor:
        # (first spatial dim is plane)
        # self.input_tensor = scn.InputLayer(dimension=3, spatial_size=[3,640,1024])
        self.input_tensor = scn.InputLayer(dimension=3,
            spatial_size=[3,spatial_size[0], spatial_size[1]])


        self.initial_convolution = scn.SubmanifoldConvolution(dimension=3,
            nIn         = 1,
            nOut        = params.n_initial_filters,
            filter_size = [1,5,5],
            bias        = params.bias)


        if params.growth_rate == GrowthRate.multiplicative:
            n_filters_next = 2 * params.n_initial_filters
        else:
            n_filters_next = params.n_initial_filters + params.n_initial_filters

        # Next, build out the convolution steps:

        self.net_core = SparseUNetCore(depth    = params.depth,
                                       inplanes = params.n_initial_filters,
                                       params   = params)

        # We need final output shaping too.
        # Even with shared weights, keep this separate:



        self.final_layer = SparseBlockSeries(inplanes = params.n_initial_filters,
                                             n_blocks = params.blocks_final,
                                             params   = params)
        #
        # self.bottleneck  = scn.SubmanifoldConvolution(dimension   = 3,
        #                                               nIn         = params.n_initial_filters,
        #                                               nOut        = 3,
        #                                               filter_size = [1,1,1],
        #                                               bias        = params.bias)

        self.bottleneck = scn.SubmanifoldConvolution(dimension=3,
            nIn         = params.n_initial_filters,
            nOut        = 3,
            filter_size = [1,1,1],
            bias        = params.bias)

        self._s_to_d = scn.SparseToDense(dimension=3, nPlanes = 3)

        # The rest of the final operations (reshape, softmax) are computed in the forward pass

        # # Configure initialization:
        # for m in self.modules():
        #     if isinstance(m, scn.SubmanifoldConvolution):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     if isinstance(m, scn.Deconvolution) or isinstance(m, scn.Convolution):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, scn.BatchNormalization):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # Use this tensor to shift the background label predictions for empty locations
        self.empty_voxel = torch.tensor([100., 0,0])
        self.empty_image = torch.zeros(size=[3,spatial_size[0], spatial_size[1]])
        self.empty_image[0,:,:] = 100

    def cuda(self, *args):
        torch.nn.Module.cuda(self, *args)

        self.empty_voxel = self.empty_voxel.cuda()
        self.empty_image = self.empty_image.cuda()

    def convert_to_scn(self, _input):

        return self.input_tensor(_input)


    def forward(self, _input):


        batch_size = _input[-1]

        x = self.input_tensor(_input)

        # Apply the initial convolutions:
        x = self.initial_convolution(x)


        # Apply the main unet architecture:
        x = self.net_core(x)



        # This squeezes into an image tensor, not 3D


        # Apply the final residual block to each plane:
        x = self.final_layer(x)

        x = self.bottleneck(x)


        # Shift all pixels down in the "background category:"
        # x.features -= self.empty_voxel

        # Convert the images to dense layout:
        x = self._s_to_d(x)



        # Break the images into 3 planes:
        x = torch.chunk(x, chunks=3, dim=2)
        x = [ _x.view(_x.shape[0], _x.shape[1], _x.shape[-2], _x.shape[-1]) for _x in x ]

        # Replace all of the locations that are 0 from spare to SparseToDense
        # With a very high background score:
        # x = [ _x + self.empty_image for _x in x ]



        return x
