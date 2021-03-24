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

class Block(nn.Module):

    def __init__(self, *, inplanes, outplanes, kernel = [3,3], padding=[1,1], params):
        nn.Module.__init__(self)


        self.conv = nn.Conv2d(
            in_channels  = inplanes,
            out_channels = outplanes,
            kernel_size  = kernel,
            stride       = [1, 1],
            padding      = padding,
            bias         = params.bias)

        self.do_batch_norm = params.batch_norm

        if self.do_batch_norm:
            self.bn   = nn.BatchNorm2d(outplanes)

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        if self.do_batch_norm:
            out = self.bn(out)
        out = self.relu(out)
        return out


class ResidualBlock(nn.Module):

    def __init__(self, *, inplanes, outplanes, kernel=[3,3], padding=[1,1], params):
        nn.Module.__init__(self)


        self.conv1 = nn.Conv2d(
            in_channels  = inplanes,
            out_channels = outplanes,
            kernel_size  = tuple(kernel),
            stride       = (1, 1),
            padding      = tuple(padding),
            bias         = params.bias)


        self.do_batch_norm = params.batch_norm
        if self.do_batch_norm:
            self.bn1 = nn.BatchNorm2d(outplanes)

        self.relu = nn.LeakyReLU(inplace=True)


        self.conv2 = nn.Conv2d(
            in_channels  = outplanes,
            out_channels = outplanes,
            kernel_size  = tuple(kernel),
            stride       = [1, 1],
            padding      = tuple(padding),
            bias         = params.bias)

        if self.do_batch_norm:
            self.bn2 = nn.BatchNorm2d(outplanes)

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


class ConvolutionDownsample(nn.Module):

    def __init__(self, *, inplanes, outplanes, params):

        nn.Module.__init__(self)

        self.conv = nn.Conv2d(
            in_channels  = inplanes,
            out_channels = outplanes,
            kernel_size  = [2, 2],
            stride       = [2, 2],
            padding      = [0, 0],
            bias         = params.bias)

        self.do_batch_norm = params.batch_norm
        if self.do_batch_norm:
            self.bn   = nn.BatchNorm2d(outplanes)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)

        if self.do_batch_norm:
            out = self.bn(out)
        out = self.relu(out)
        return out


class ConvolutionUpsample(nn.Module):

    def __init__(self, *, inplanes, outplanes, params):

        nn.Module.__init__(self)

        self.conv = nn.ConvTranspose2d(
            in_channels  = inplanes,
            out_channels = outplanes,
            kernel_size  = [2, 2],
            stride       = [2, 2],
            padding      = [0, 0],
            bias         = params.bias)

        self.do_batch_norm = params.batch_norm
        if self.do_batch_norm:
            self.bn   = nn.BatchNorm2d(outplanes)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):

        out = self.conv(x)

        if self.do_batch_norm:
            out = self.bn(out)
        out = self.relu(out)
        return out


class BlockSeries(torch.nn.Module):


    def __init__(self, *, inplanes, n_blocks, kernel = [3,3], padding = [1,1], params):
        torch.nn.Module.__init__(self)


        self.blocks = torch.nn.ModuleList()
        if not params.residual:
            for i in range(n_blocks):
                self.blocks.append(Block(
                                inplanes  = inplanes,
                                outplanes = inplanes,
                                kernel    = kernel,
                                padding   = padding,
                                params    = params))
        else:
            for i in range(n_blocks):
                self.blocks.append(ResidualBlock(
                                inplanes  = inplanes,
                                outplanes = inplanes,
                                kernel    = kernel,
                                padding   = padding,
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

            kernel  = (params.filter_size_deepest, params.filter_size_deepest)
            padding = tuple( int((k - 1) / 2) for k in kernel )

            self.blocks = BlockSeries(
                inplanes    = n_filters_bottleneck,
                kernel      = kernel,
                padding     = padding,
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

            kernel  = (params.filter_size_deepest, params.filter_size_deepest)
            padding = tuple( int((k - 1) / 2) for k in kernel )

            self.blocks = BlockSeries(
                inplanes    = n_filters_bottleneck,
                kernel      = kernel,
                padding     = padding,
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
        else:

            x = torch.cat(x, dim=1)
            x = self.bottleneck(x)
            x = self.blocks(x)
            x = self.unbottleneck(x)
            x = torch.chunk(x, chunks=3, dim=1)


        return x

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

        if depth == 0:
            self.main_module = DeepestBlock(inplanes = inplanes,
                                            params = params)
        else:
            # Residual or convolutional blocks, applied in series:
            self.down_blocks = BlockSeries(inplanes = inplanes,
                                           n_blocks = self.layers,
                                           params   = params)

            if params.growth_rate == "multiplicative":
                n_filters_next = 2 * inplanes
            else:
                n_filters_next = inplanes + params.n_initial_filters

            # Down sampling operation:
            # This does change the number of filters from above down-pass blocks
            if params.downsampling == "convolutional":
                self.downsample = ConvolutionDownsample(inplanes    = inplanes,
                                                        outplanes   = n_filters_next,
                                                        params      = params)
            else:
                self.downsample = MaxPooling(inplanes    = inplanes,
                                             outplanes   = n_filters_next,
                                             params      = params)



            # Submodule:
            self.main_module    = UNetCore(depth    = depth-1,
                                           inplanes = n_filters_next,
                                           params   = params )

            # Upsampling operation:
            if params.upsampling == "convolutional":
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
            if params.connections == "sum":
                self.connection = SumConnection()
            elif params.connections == "concat":
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
        x = self.main_module(x)

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



        #
        # if FLAGS.VERBOSITY >1:
        #     for p in range(len(x)):
        #         print("plane {} Depth {}, shape: ".format(p, self.depth), x[p].shape)

        return x




class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


class UResNet(torch.nn.Module):

    def __init__(self, params):


        torch.nn.Module.__init__(self)


        self.initial_convolution = Block(
            inplanes  = 1,
            kernel    = [5,5],
            padding   = [2,2],
            outplanes = params.n_initial_filters,
            params    = params)

        n_filters = params.n_initial_filters

        self.net_core = UNetCore(
            depth    = params.network_depth,
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



        # Reshape this tensor into the right shape to apply this multiplane network.
        x = input_tensor

        x = torch.chunk(x, chunks=3, dim=1)


        # Apply the initial convolutions:
        x = tuple( self.initial_convolution(_x) for _x in x )



        #
        # if VERBOSITY >1:
        #     print("After Initial convolution: ")
        #     for p in [0,1,2]:
        #         print("Plane {}, shape ".format(p), x[p].shape)

        # Apply the main unet architecture:
        x = self.net_core(x)

        # Apply the final residual block to each plane:
        x = tuple( self.final_layer(_x) for _x in x )
        x = tuple( self.bottleneck(_x) for _x in x )
        # Might need to do some reshaping here
        return x
