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
from src.config.network import DownSampling, UpSampling
from src.config.network import Connection, GrowthRate, Norm
from src.config.network import BlockStyle

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

        self._do_normalization = False
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


class ConvNextBlock(nn.Module):

    def __init__(self, *, inplanes, outplanes, nplanes=1, params):

        nn.Module.__init__(self)


        kernel_1  = [nplanes, params.kernel_size,params.kernel_size]


        self.conv1 = scn.SubmanifoldConvolution(dimension=3,
            nIn         = inplanes,
            nOut        = outplanes,
            filter_size = kernel_1,
            bias        = params.bias)

        if params.normalization == Norm.batch:
            self.norm = scn.BatchNorm(outplanes)
        elif params.normalization == Norm.layer:
            raise Exception("Layer norm not supported in SCN")
        else:
            self.norm = lambda x: x

        self.conv2 = scn.SubmanifoldConvolution(dimension=3,
            nIn         = inplanes,
            nOut        = 4*inplanes,
            filter_size = [nplanes,1,1],
            bias        = params.bias)

        self.activation = scn.LeakyReLU()

        self.conv3 = scn.SubmanifoldConvolution(dimension=3,
            nIn         = 4*inplanes,
            nOut        = inplanes,
            filter_size = [nplanes,1,1],
            bias        = params.bias)
        

        self.residual = scn.Identity()

        self.add = scn.AddTable()


    def forward(self, x):

        residual = self.residual(x)

        out = self.conv1(x)

        out = self.norm(out)

        out = self.conv2(out)

        out = self.activation(out)

        out = self.conv3(out)

        out = self.add([out, residual])

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

        if params.block_style == BlockStyle.none:
            self.blocks = [ SparseBlock(
                                inplanes  = inplanes,
                                outplanes = inplanes,
                                nplanes   = n_planes,
                                params    = params)
                            for i in range(n_blocks)]
        elif params.block_style == BlockStyle.residual:
            self.blocks = [ SparseResidualBlock(
                                inplanes  = inplanes,
                                outplanes = inplanes,
                                nplanes   = n_planes,
                                params    = params)
                            for i in range(n_blocks)
                ]
        elif params.block_style == BlockStyle.convnext:
            self.blocks = [ ConvNextBlock(
                                inplanes  = inplanes,
                                outplanes = inplanes,
                                nplanes   = n_planes,
                                params    = params)
                            for i in range(n_blocks)
                ]
            
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

        x = self.merger(x)
        x = self.blocks(x)
        x = self.splitter(x)

        classification_head = x


        return x, classification_head, None


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
        self.vertex_depth = params.vertex.depth

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
        x, classification_head, vertex_head = self.main_module(x)

        if self.depth != 0:


            # perform the upsampling step:
            # perform the downsampling operation:
            x = self.upsample(x)

            # Connect with the residual if necessary:
            x = self.connection(x, residual)

            # Apply the convolutional steps:
            x = self.up_blocks(x)

        if self.depth == self.vertex_depth: vertex_head = x


        return x, classification_head, vertex_head



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


        if params.classification.active:

            # The image size here is going to be the orignal / 2**depth
            # We need to know it for the pooling layer
            pool_size = [d // 2**params.depth for d in spatial_size]

            n_filters = params.n_initial_filters
            for i in range(params.depth):
                if params.growth_rate == GrowthRate.multiplicative:
                    n_filters = 2 * n_filters
                else:
                    n_filters = n_filters + params.n_initial_filters


            self.classifier_input = scn.Convolution(
                dimension   = 3,
                nIn         = n_filters,
                nOut        = params.classification.n_filters,
                filter_size = [3,1,1],
                filter_stride = [1,1,1],
                bias        = params.bias)

            self.classifier = SparseBlockSeries(
                inplanes = params.classification.n_filters,
                n_blocks = params.classification.n_layers,
                params   = params
            )

            self.bottleneck_classifier = scn.SubmanifoldConvolution(
                dimension   = 3,
                nIn         = params.classification.n_filters,
                nOut        = 4,
                filter_size = [1,1,1],
                # filter_stride = [1,1,1],
                bias        = params.bias)

            self.pool = scn.AveragePooling(
                dimension   = 3,
                pool_size   = [1,] + pool_size,
                pool_stride = [1,1,1],
            )

            self.classifier_sparse_to_dense = scn.SparseToDense(dimension=3, nPlanes=4)


        if params.vertex.active:

            vertex_size = [ d // 2**(params.depth - params.vertex.depth ) for d in spatial_size]
           
            n_filters = params.n_initial_filters
            for i in range(params.depth - params.vertex.depth):
                if params.growth_rate == GrowthRate.multiplicative:
                    n_filters = 2 * n_filters
                else:
                    n_filters = n_filters + params.n_initial_filters

            self.vertex_layers = SparseBlockSeries(
                inplanes  = n_filters,
                n_blocks  = params.vertex.n_layers,
                params    = params
            )

            self.bottleneck_vertex = scn.SubmanifoldConvolution(
                dimension   = 3,
                nIn         = n_filters,
                nOut        = 3,
                filter_size = [1,1,1],
                bias        = True
            )

            self.vertex_sigmoid = scn.Sigmoid()

            self.vertex_sparse_to_dense = scn.SparseToDense(dimension=3, nPlanes=3)


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
        # self.empty_voxel = torch.tensor([100., 0,0])
        # self.empty_image = torch.zeros(size=[3,spatial_size[0], spatial_size[1]])
        # self.empty_image[0,:,:] = 100

    def cuda(self, *args):
        torch.nn.Module.cuda(self, *args)
        #
        # self.empty_voxel = self.empty_voxel.cuda()
        # self.empty_image = self.empty_image.cuda()

    def convert_to_scn(self, _input):

        return self.input_tensor(_input)


    def forward(self, _input):


        batch_size = _input[-1]
        return_dict = {
            "event_label" : None,
            "vertex"      : None,
        }

        x = self.input_tensor(_input)
        init_size = x.spatial_size

        # Apply the initial convolutions:
        x = self.initial_convolution(x)


        # Apply the main unet architecture:
        seg_labels, classification_head, vertex_head = self.net_core(x)

        # This squeezes into an image tensor, not 3D

        # Apply the final residual block to each plane:
        seg_labels = self.final_layer(seg_labels)

        seg_labels = self.bottleneck(seg_labels)


        # Shift all pixels down in the "background category:"
        # seg_labels.features -= self.empty_voxel

        # Convert the images to dense layout:
        seg_labels = self._s_to_d(seg_labels)



        # Break the images into 3 planes:
        seg_labels = torch.chunk(seg_labels, chunks=3, dim=2)
        seg_labels = [ _x.view(_x.shape[0], _x.shape[1], _x.shape[-2], _x.shape[-1]) for _x in seg_labels ]

        return_dict["segmentation"] = seg_labels

        if hasattr(self, "classifier"):
            classification_head = classification_head.detach()
            classified = self.classifier_input(classification_head)
            classified = self.classifier(classified)
            classified = self.bottleneck_classifier(classified)
            # 4 classes of events:
            classified = self.pool(classified)
            classified = self.classifier_sparse_to_dense(classified)
            classified = classified.view(classified.shape[0], classified.shape[1])
            return_dict["event_label"] = classified


        if hasattr(self, "vertex_layers"):
            vertex_head = vertex_head.detach()
            vertex = self.vertex_layers(vertex_head)
            vertex = self.bottleneck_vertex(vertex)

            # Apply a sigmoid before going to dense:
            vertex = self.vertex_sigmoid(vertex)

            vertex = self.vertex_sparse_to_dense(vertex)

            # Split into per-plane results:
            vertex = torch.chunk(vertex, chunks=3, dim=2)
            vertex = [ _x.view(_x.shape[0], _x.shape[1], _x.shape[-2], _x.shape[-1]) for _x in vertex ]

            return_dict["vertex"] = vertex

        return return_dict  
