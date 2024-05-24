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

from src.config.network import Connection, GrowthRate, BlockStyle
from src.config.network import DownSampling, UpSampling, Norm

activation_function = nn.functional.leaky_relu

import copy

# This is a work around to avoid an issue on XPU:
class InstanceNorm2d(torch.nn.Module):

    def __init__(self, num_features):
        nn.Module.__init__(self)


    def forward(self, x):
        x_mean = torch.mean(x, axis=(-2,-1), keepdims=True)
        x_var = torch.var(x, axis=(-2,-1), keepdims=True, correction=0)
        x_norm = (x - x_mean) / torch.sqrt(x_var + 1e-5)

        return x_norm

class Block(nn.Module):

    def __init__(self, *,
            inplanes,
            outplanes,
            strides    = None,
            padding    = None,
            kernel     = None,
            activation = activation_function,
            override_norm = None,
            groups     = None,
            params):
        nn.Module.__init__(self)

        if kernel is None:
            kernel = [params.kernel_size,params.kernel_size]
        if padding is None:
            padding = tuple( int((k - 1) / 2) for k in kernel )
        if strides is None:
            strides = [1,1 ]

        if groups is None:
            groups = 1

        self.conv = nn.Conv2d(
            in_channels  = inplanes,
            out_channels = outplanes,
            kernel_size  = kernel,
            stride       = strides,
            padding      = padding,
            groups       = groups,
            bias         = params.bias)

        norm = params.normalization if override_norm is not None else override_norm

        if params.normalization == Norm.batch:
            self._do_normalization = True
            self.norm = nn.BatchNorm2d(outplanes)
        elif params.normalization == Norm.group:
            self._do_normalization = True
            self.norm = nn.GroupNorm(num_groups=4, num_channels=outplanes)
        elif params.normalization == Norm.layer:
            self._do_normalization = True
            self.norm = "layer"
        elif params.normalization == Norm.instance:
            self._do_normalization = True
            self.norm = InstanceNorm2d(outplanes)
        else:
            self._do_normalization = False


        self.activation = activation

    def forward(self, x):

        out = self.conv(x)

        if self._do_normalization:
            if self.norm == "layer":
                norm_shape = out.shape[1:]
                # norm_shape = torch.tensor([8,] + list(norm_shape)).to(x.device)
                self.norm = torch.nn.LayerNorm(normalized_shape=norm_shape)
                self.norm.to(out.device)
                # out = torch.nn.functional.layer_norm(out, norm_shape)
                out = self.norm(out)
            else:
                out = self.norm(out)
        out = self.activation(out)
        return out


class ResidualBlock(nn.Module):

    def __init__(self, *, inplanes, outplanes, params):
        nn.Module.__init__(self)

        kernel = [params.kernel_size,params.kernel_size]
        padding = tuple( int((k - 1) / 2) for k in kernel )

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

        out = self.convolution_2(out)


        out += residual
        out = activation_function(out)

        return out

class ConvNextBlock(nn.Module):



    def __init__(self, *, inplanes, outplanes, params):

        nn.Module.__init__(self)


        kernel_1  = [params.kernel_size,params.kernel_size]
        padding_1 = tuple( int((k - 1) / 2) for k in kernel_1 )

        if params.depthwise:
            groups = inplanes
        else:
            groups = 1

        self.convolution_1 = Block(
            inplanes    = inplanes,
            outplanes   = inplanes,
            kernel      = kernel_1,
            padding     = padding_1,
            groups      = groups,
            activation  = torch.nn.Identity(),
            params      = params)


        kernel_2  = [1, 1]
        padding_2 = [0, 0]



        self.convolution_2 = Block(
            inplanes    = inplanes,
            outplanes   = 4 * inplanes,
            kernel      = kernel_2,
            padding     = padding_2,
            override_norm = "none", # Sets to no normalization, as opposed to the default param
            params      = params)

        self.convolution_3 = Block(
            inplanes    = 4 * inplanes,
            outplanes   = outplanes,
            kernel      = kernel_2,
            padding     = padding_2,
            activation  = torch.nn.Identity(),
            override_norm = "", # Sets to no normalization, as opposed to the default param
            params      = params)


    def forward(self, x):

        residual = x

        out = self.convolution_1(x)

        out = self.convolution_2(out)

        out = self.convolution_3(out)

        out += residual

        return out
    
class ConvolutionUpsample(nn.Module):

    def __init__(self, *, inplanes, outplanes, activation=activation_function, params):

        nn.Module.__init__(self)

        self.conv = nn.ConvTranspose2d(
            in_channels  = inplanes,
            out_channels = outplanes,
            kernel_size  = [2, 2],
            stride       = [2, 2],
            padding      = [0, 0],
            bias         = params.bias)


        if params.normalization == Norm.batch:
            self._do_normalization = True
            self.norm = nn.BatchNorm2d(outplanes)
        elif params.normalization == Norm.group:
            self._do_normalization = True
            self.norm = nn.GroupNorm(num_groups=4, num_channels=outplanes)
        elif params.normalization == Norm.layer:
            self._do_normalization = True
            self.norm = "layer"
            # Have to do something special here to avoid pre-computing all the shapes ...
        elif params.normalization == Norm.instance:
            self._do_normalization = True
            self.norm = InstanceNorm2d(outplanes)
        else:
            self._do_normalization = False


        self.activation = activation

    def forward(self, x):


        if self._do_normalization:
            if self.norm == "layer":
                norm_shape = x.shape[1:]
                # norm_shape[0] = 8
                x = torch.nn.functional.layer_norm(x, norm_shape)
            else:
                x = self.norm(x)
        
        out = self.conv(x)

        out = self.activation(out)
        return out


class BlockSeries(torch.nn.Module):


    def __init__(self, *, inplanes, n_blocks, params):
        torch.nn.Module.__init__(self)


        self.blocks = torch.nn.ModuleList()

        if params.block_style == BlockStyle.none:
            for i in range(n_blocks):
                self.blocks.append(Block(
                                inplanes  = inplanes,
                                outplanes = inplanes,
                                params    = params))
        elif params.block_style == BlockStyle.residual:
            for i in range(n_blocks):
                self.blocks.append(ResidualBlock(
                                inplanes  = inplanes,
                                outplanes = inplanes,
                                params    = params))
        elif params.block_style == BlockStyle.convnext:
            for i in range(n_blocks):
                self.blocks.append(ConvNextBlock(
                                inplanes  = inplanes,
                                outplanes = inplanes,
                                params    = params))


    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

        return x


class DeepestBlock(nn.Module):

    def __init__(self, *, inplanes, params):
        nn.Module.__init__(self)


        # The deepest block concats across planes, applies convolutions,
        # Then splits into planes again


        # #  It's possible to split this and prevent concating as a test.
        # if FLAGS.BLOCK_CONCAT:
        #     self.blocks = BlockSeries(inplanes, blocks_deepest_layer, residual = residual,
        #         use_bias=use_bias, batch_norm=batch_norm)
        # else:
        n_filters_bottleneck = params.bottleneck_deepest

        self.block_concat = params.block_concat

        if self.block_concat:


            self.bottleneck = Block(
                    inplanes   = inplanes,
                    outplanes  = n_filters_bottleneck,
                    kernel     = [1,1],
                    padding    = [0,0],
                    params     = params)

            # kernel = [ params.kernel_size, params.kernel_size ]
            # padding = tuple( int((k - 1) / 2) for k in kernel )

            self.blocks = BlockSeries(
                inplanes    = n_filters_bottleneck,
                n_blocks    = params.blocks_deepest_layer,
                params      = params)

            self.unbottleneck = Block(
                    inplanes   = n_filters_bottleneck,
                    outplanes  = inplanes,
                    kernel     = [1,1],
                    padding    = [0,0],
                    params     = params)


        else:

            self.bottleneck = Block(
                    inplanes   = 3*inplanes,
                    outplanes  = n_filters_bottleneck,
                    kernel     = [1,1],
                    padding    = [0,0],
                    params     = params)
            

            self.blocks = BlockSeries(
                inplanes    = n_filters_bottleneck,
                n_blocks    = params.blocks_deepest_layer,
                params      = params)

            self.unbottleneck = Block(
                    inplanes   = n_filters_bottleneck,
                    outplanes  = 3*inplanes,
                    kernel     = [1,1],
                    padding    = [0,0],
                    params     = params)



    def forward(self, x):

        # THis isn't really a recommended setting to use, but we can control whether or not to connect here:
        # if FLAGS.BLOCK_CONCAT:
        #     x = [ self.blocks(_x) for _x in x ]
        # else:
        if self.block_concat:
            x = ( self.bottleneck(_x) for _x in x )
            x = ( self.blocks(_x) for _x in x )
            x = ( self.unbottleneck(_x) for _x in x )
            classification_head = torch.cat(x, dim=1)
        else:

            x = torch.cat(x, dim=1)
            x = self.bottleneck(x)
            x = self.blocks(x)
            x = self.unbottleneck(x)
            classification_head = x
            x = torch.chunk(x, chunks=3, dim=1)


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

    def __init__(self, *, inplanes, params):
        nn.Module.__init__(self)

        self.bottleneck = Block(
            inplanes    = 2*inplanes,
            outplanes   = inplanes,
            kernel      = [1,1],
            padding     = [0,0],
            params      = params)
        # self.bottleneck = nn.Conv2d(
        #     in_channels   = 2*inplanes,
        #     out_channels  = inplanes,
        #     kernel_size   = 1,
        #     stride        = 1,
        #     padding       = 0,
        #     bias          = params.bias)

    def forward(self, x, residual):
        x = torch.cat((x, residual), dim=1)
        x = self.bottleneck(x)
        return x




class MaxPooling(nn.Module):

    def __init__(self,*, inplanes, outplanes, params):
        nn.Module.__init__(self)


        self.pool = torch.nn.MaxPool2d(stride=2, kernel_size=2)

        self.bottleneck = Block(
            inplanes    = inplanes,
            outplanes   = outplanes,
            kernel      = (1,1),
            padding     = (0,0),
            params      = params)

    def forward(self, x):
        x = self.pool(x)

        return self.bottleneck(x)

class InterpolationUpsample(nn.Module):

    def __init__(self, *, inplanes, outplanes, params):
        nn.Module.__init__(self)


        self.up = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.bottleneck = Block(
            inplanes    = inplanes,
            outplanes   = outplanes,
            kernel      = (1,1),
            padding     = (0,0),
            params      = params)

    def forward(self, x):
        x = self.up(x)
        return self.bottleneck(x)


class UNetCore(nn.Module):

    def __init__(self, *,  depth, inplanes,  params):

        nn.Module.__init__(self)


        self.layers = params.blocks_per_layer
        self.depth  = depth
        self.vertex_depth = params.vertex.depth


        if depth == 0:
            self.main_module = DeepestBlock(inplanes = inplanes,
                                            params = params)
        else:
            # Residual or convolutional blocks, applied in series:
            self.down_blocks = BlockSeries(inplanes = inplanes,
                                           n_blocks = self.layers,
                                           params   = params)

            if params.growth_rate == GrowthRate.multiplicative:
                n_filters_next = 2 * inplanes
            else:
                n_filters_next = inplanes + params.n_initial_filters

            # Down sampling operation:
            # This does change the number of filters from above down-pass blocks
            if params.downsampling == DownSampling.convolutional:
                self.downsample = Block(
                    inplanes    = inplanes,
                    outplanes   = n_filters_next,
                    strides     = (2,2),
                    padding     = (0,0),
                    kernel      = (2,2),
                    params      = params)
                # self.downsample = ConvolutionDownsample(inplanes    = inplanes,
                #                                         outplanes   = n_filters_next,
                #                                         params      = params)
            else:
                self.downsample = MaxPooling(inplanes    = inplanes,
                                             outplanes   = n_filters_next,
                                             params      = params)



            # Submodule:
            self.main_module    = UNetCore(depth    = depth-1,
                                           inplanes = n_filters_next,
                                           params   = params )

            # Upsampling operation:
            if params.upsampling == UpSampling.convolutional:
                self.upsample       = ConvolutionUpsample(inplanes  = n_filters_next,
                                                          outplanes = inplanes,
                                                          params    = params)
            else:
                self.upsample = InterpolationUpsample(inplanes  = n_filters_next,
                                                      outplanes = inplanes,
                                                      params    = params)


            # Convolutional or residual blocks for the upsampling pass:
            self.up_blocks = BlockSeries(inplanes = inplanes,
                                         n_blocks = self.layers,
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

            residual = x

            x = tuple( self.down_blocks(_x) for _x in x )

            # perform the downsampling operation:
            x = tuple( self.downsample(_x) for _x in x )
        #
        # if FLAGS.VERBOSITY >1:
        #     for p in range(len(x)):
        #         print("plane {} Depth {}, shape: ".format(p, self.depth), x[p].shape)


        # Apply the main module:
        x, classification_head, vertex_head = self.main_module(x)

        # The vertex_head is None after the DEEPEST layer.  But, if we're returning it, do it here:


        if self.depth != 0:

            # perform the upsampling step:
            # perform the downsampling operation:
            x = tuple( self.upsample(_x) for _x in x )

            # Connect with the residual if necessary:
            # for i in range(len(x)):
            #     x[i] = self.connection(x[i], residual=residual[i])

            x = tuple( self.connection(_x, _r) for _x, _r in zip(x, residual))


            # Apply the convolutional steps:
            x = tuple( self.up_blocks(_x) for _x in x )

        if self.depth == self.vertex_depth: vertex_head = x


        return x, classification_head, vertex_head



class UResNet(torch.nn.Module):

    def __init__(self, params, spatial_size):

        torch.nn.Module.__init__(self)

        self.initial_convolution = Block(
            inplanes  = 1,
            kernel    = [5,5],
            padding   = [2,2],
            outplanes = params.n_initial_filters,
            params    = params)

        n_filters = params.n_initial_filters

        self.net_core = UNetCore(
            depth    = params.depth,
            inplanes = params.n_initial_filters,
            params   = params )

        # We need final output shaping too.

        self.final_layer = BlockSeries(
            inplanes = params.n_initial_filters,
            n_blocks = params.blocks_final,
            params   = params )


        self.bottleneck = nn.Conv2d(
            in_channels  = params.n_initial_filters,
            out_channels = 3,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            bias         = params.bias)

        # The rest of the final operations (reshape, softmax) are computed in the forward pass
        if params.classification.active:

            # The image size here is going to be the orignal / 2**depth
            # We need to know it for the pooling layer
            self.pool_size = [d // 2**params.depth for d in spatial_size]

            n_filters = params.n_initial_filters
            for i in range(params.depth):
                if params.growth_rate == GrowthRate.multiplicative:
                    n_filters = 2 * n_filters
                else:
                    n_filters = n_filters + params.n_initial_filters

            self.classification_detach = params.classification.detach

            n_filters = 3*n_filters
            self.classifier_input = nn.Conv2d(
                in_channels  = n_filters,
                out_channels = params.classification.n_filters,
                kernel_size  = 1,
                stride       = 1,
                padding      = 0,
                bias         = params.bias
            )

            self.classifier = BlockSeries(
                inplanes = params.classification.n_filters,
                n_blocks = params.classification.n_layers,
                params   = params
            )
            self.bottleneck_classifer = nn.Conv2d(
                in_channels  = params.classification.n_filters,
                out_channels = 4,
                kernel_size  = 1,
                stride       = 1,
                padding      = 0,
                bias         = params.bias)

            self.pool = torch.nn.AvgPool2d(self.pool_size)

        if params.vertex.active:

            vertex_size = [ d // 2**(params.depth - params.vertex.depth ) for d in spatial_size]

            n_filters = params.n_initial_filters
            for i in range(params.depth - params.vertex.depth):
                if params.growth_rate == GrowthRate.multiplicative:
                    n_filters = 2 * n_filters
                else:
                    n_filters = n_filters + params.n_initial_filters

            self.vertex_detach = params.vertex.detach

            self.vertex_input = nn.Conv2d(
                in_channels  = n_filters,
                out_channels = params.vertex.n_filters,
                kernel_size  = 1,
                stride       = 1,
                padding      = 0,
                bias         = params.bias)

            self.vertex_layers = BlockSeries(
                inplanes  = params.vertex.n_filters,
                n_blocks  = params.vertex.n_layers,
                params    = params
            )

            self.bottleneck_vertex = nn.Conv2d(
                in_channels  = params.vertex.n_filters,
                out_channels = 3,
                kernel_size  = 1,
                stride       = 1,
                padding      = 0,
                bias         = params.bias)

        #
        # Configure initialization:
        for m in self.modules():
            # if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                # nn.init.xavier_uniform_(m.bias.data)
                if params.bias:
                    nn.init.constant_(m.bias.data,0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_tensor):


        batch_size = input_tensor.shape[0]

        return_dict = {
            "event_label" : None,
            "vertex"      : None,
        }


        # Reshape this tensor into the right shape to apply this multiplane network.
        x = input_tensor


        x = torch.chunk(x, chunks=3, dim=1)

        # Apply the initial convolutions:
        x = tuple( self.initial_convolution(_x) for _x in x )



        # Apply the main unet architecture:
        seg_labels, classification_head, vertex_head = self.net_core(x)

        # Apply the final residual block to each plane:
        seg_labels = tuple( self.final_layer(_x) for _x in seg_labels )
        seg_labels = tuple( self.bottleneck(_x) for _x in seg_labels )

        # Always return the segmentation
        return_dict["segmentation"] = seg_labels

        if hasattr(self, "classifier"):
            if self.classification_detach:
                classification_head = classification_head.detach()
            classified = self.classifier_input(classification_head)
            classified = self.classifier(classified)
            classified = self.bottleneck_classifer(classified)
            # 4 classes of events:
            classified = self.pool(classified).reshape((-1, 4))
            return_dict["event_label"] = classified


        if hasattr(self, "vertex_layers"):
            if self.vertex_detach:
                vertex_head = [ v.detach() for v in vertex_head ]
            vertex = [ self.vertex_input(v) for v in vertex_head ]
            vertex = [ self.vertex_layers(v) for v in vertex ]
            vertex = [ self.bottleneck_vertex(v) for v in vertex ]

            # Apply a sigmoid before returning:

            return_dict["vertex"] = [ torch.sigmoid(v) for v in vertex ]


        return return_dict
