import torch
import torch.nn as nn


from src import utils
FLAGS = utils.flags.FLAGS()

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

    def __init__(self, inplanes, outplanes):
        nn.Module.__init__(self)


        self.conv = nn.Conv2d(
            in_channels  = inplanes, 
            out_channels = outplanes, 
            kernel_size  = [3, 3],
            stride       = [1, 1],
            padding      = [1, 1],
            bias         = FLAGS.USE_BIAS)

        if FLAGS.BATCH_NORM:
            self.bn   = nn.BatchNorm2d(outplanes)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        if FLAGS.BATCH_NORM:
            out = self.bn(out)
        out = self.relu(out)
        return out


class ResidualBlock(nn.Module):

    def __init__(self, inplanes, outplanes):
        nn.Module.__init__(self)


        self.conv1 = nn.Conv2d(
            in_channels  = inplanes, 
            out_channels = outplanes, 
            kernel_size  = [3, 3],
            stride       = [1, 1],
            padding      = [1, 1],
            bias         = FLAGS.USE_BIAS)

        if FLAGS.BATCH_NORM:
            self.bn1 = nn.BatchNorm2d(outplanes)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(
            in_channels  = outplanes, 
            out_channels = outplanes, 
            kernel_size  = [3, 3],
            stride       = [1, 1],
            padding      = [1, 1],
            bias         = FLAGS.USE_BIAS)

        if FLAGS.BATCH_NORM:
            self.bn2 = nn.BatchNorm2d(outplanes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if FLAGS.BATCH_NORM:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        if FLAGS.BATCH_NORM:
            out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class ConvolutionDownsample(nn.Module):

    def __init__(self, inplanes, outplanes):
        nn.Module.__init__(self)

        self.conv = nn.Conv2d(
            in_channels  = inplanes, 
            out_channels = outplanes, 
            kernel_size  = [2, 2],
            stride       = [2, 2],
            padding      = [0, 0],
            bias         = FLAGS.USE_BIAS)
        if FLAGS.BATCH_NORM:
            self.bn   = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        if FLAGS.BATCH_NORM:
            out = self.bn(out)
        out = self.relu(out)
        return out


class ConvolutionUpsample(nn.Module):

    def __init__(self, inplanes, outplanes):
        nn.Module.__init__(self)

        self.conv = nn.ConvTranspose2d(
            in_channels  = inplanes, 
            out_channels = outplanes, 
            kernel_size  = [2, 2],
            stride       = [2, 2],
            padding      = [0, 0],
            bias         = FLAGS.USE_BIAS)
        if FLAGS.BATCH_NORM:
            self.bn   = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.conv(x)
        if FLAGS.BATCH_NORM:
            out = self.bn(out)
        out = self.relu(out)
        return out


class BlockSeries(torch.nn.Module):


    def __init__(self, inplanes, n_blocks, residual):
        torch.nn.Module.__init__(self)

        if not residual:
            self.blocks = [ Block(inplanes, inplanes) for i in range(n_blocks) ]
        else:
            self.blocks = [ ResidualBlock(inplanes, inplanes) for i in range(n_blocks)]

        for i, block in enumerate(self.blocks):
            self.add_module('block_{}'.format(i), block)


    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

        return x


class DeepestBlock(nn.Module):

    def __init__(self, inplanes, residual):
        nn.Module.__init__(self)
        

        # The deepest block concats across planes, applies convolutions,
        # Then splits into planes again


        #  It's possible to split this and prevent concating as a test.
        if FLAGS.BLOCK_CONCAT:
            self.blocks = BlockSeries(inplanes, FLAGS.RES_BLOCKS_DEEPEST_LAYER, residual = residual)
        else:
            self.blocks = BlockSeries(FLAGS.NPLANES * inplanes, FLAGS.RES_BLOCKS_DEEPEST_LAYER, residual = residual)



    def forward(self, x):
        

        # THis isn't really a recommended setting to use, but we can control whether or not to connect here:
        if FLAGS.BLOCK_CONCAT:
            x = [ self.blocks(_x) for _x in x ]
        else:
            x = torch.cat(x, dim=1)
            x = self.blocks(x)
            x = torch.chunk(x, chunks=FLAGS.NPLANES, dim=1)


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
    
    def __init__(self, inplanes):
        nn.Module.__init__(self)

        self.bottleneck = nn.Conv2d(
            in_channels   = 2*inplanes, 
            out_channels  = inplanes, 
            kernel_size   = 1,
            stride        = 1,
            padding       = 0,
            bias          = False)

    def forward(self, x, residual):
        x = torch.cat([x, residual], dim=1)
        return self.bottleneck(x)


class UNetCore(nn.Module):

    def __init__(self, depth, nlayers, inplanes, residual):
        nn.Module.__init__(self)
        

        self.layers = nlayers
        self.depth  = depth

        if depth == 1:
            self.main_module = DeepestBlock(inplanes, residual = residual)
        else:
            # Residual or convolutional blocks, applied in series:
            self.down_blocks = BlockSeries(inplanes, nlayers, residual=residual)

            if FLAGS.GROWTH_RATE == "linear":
                n_filters_next_layer = inplanes + FLAGS.N_INITIAL_FILTERS
            elif FLAGS.GROWTH_RATE == "multiplicative":
                n_filters_next_layer = inplanes * 2

            # Down sampling operation:
            self.downsample     = ConvolutionDownsample(inplanes, n_filters_next_layer)
            
            
            # Submodule: 
            self.main_module    = UNetCore(depth-1, nlayers, n_filters_next_layer, residual = residual)
            # Upsampling operation:
            self.upsample       = ConvolutionUpsample(n_filters_next_layer, inplanes)


            # Convolutional or residual blocks for the upsampling pass:
            self.up_blocks = BlockSeries(inplanes, nlayers, residual=residual)

            # Residual connection operation:
            if FLAGS.CONNECTIONS == "sum":
                self.connection = SumConnection()
            elif FLAGS.CONNECTIONS == "concat":
                self.connection = ConcatConnection(inplanes)
            else:
                self.connection = NoConnection()


    def forward(self, x):
        
        # Take the input and apply the downward pass convolutions.  Save the residual
        # at the correct time.
        if self.depth != 1:
            if FLAGS.CONNECT_PRE_RES_BLOCKS_DOWN:
                residual = x

            x = [ self.down_blocks(_x) for _x in x ]
            
            if not FLAGS.CONNECT_PRE_RES_BLOCKS_DOWN:
                residual = x

            # perform the downsampling operation:
            x = [ self.downsample(_x) for _x in x ]

        # Apply the main module:
        x = self.main_module(x)


        if self.depth != 1:

            # perform the upsampling step:
            # perform the downsampling operation:
            x = [ self.upsample(_x) for _x in x ]

            # Connect with the residual if necessary:
            if FLAGS.CONNECT_PRE_RES_BLOCKS_UP:
                for i in range(len(x)):
                    x[i] = self.connection(x[i], residual=residual[i])


            # Apply the convolutional steps:
            x = [ self.up_blocks(_x) for _x in x ]
                
            if not FLAGS.CONNECT_PRE_RES_BLOCKS_UP:
                for i in range(len(x)):

                    x[i] = self.connection(x[i], residual=residual[i])

        return x





class UResNet(torch.nn.Module):

    def __init__(self, shape):
        torch.nn.Module.__init__(self)


        

        self.initial_convolution = nn.Conv2d(
            in_channels  = 1,
            out_channels = FLAGS.N_INITIAL_FILTERS,
            kernel_size  = [5, 5], 
            stride       = [1, 1],
            padding      = [2, 2], 
            bias         = FLAGS.USE_BIAS)

        n_filters = FLAGS.N_INITIAL_FILTERS
        # Next, build out the convolution steps:

        self.net_core = UNetCore(
            depth    = FLAGS.NETWORK_DEPTH, 
            nlayers  = FLAGS.RES_BLOCKS_PER_LAYER,
            inplanes = FLAGS.N_INITIAL_FILTERS,
            residual = FLAGS.RESIDUAL)

        # We need final output shaping too.  
        # Even with shared weights, keep this separate:

        self.final_layer = BlockSeries(
            FLAGS.N_INITIAL_FILTERS, 
            FLAGS.RES_BLOCKS_FINAL, 
            residual=FLAGS.RESIDUAL) 

        self.bottleneck = nn.Conv2d(
            in_channels  = FLAGS.N_INITIAL_FILTERS, 
            out_channels = 3, 
            kernel_size  = 1, 
            stride       = 1,
            padding      = 0,
            bias         = False)

        # The rest of the final operations (reshape, softmax) are computed in the forward pass


        # Configure initialization:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_tensor):
        
        
        batch_size = input_tensor.shape[0]



        # Reshape this tensor into the right shape to apply this multiplane network.
        x = input_tensor

        x = torch.chunk(x, chunks=FLAGS.NPLANES, dim=1)



        # Apply the initial convolutions:
        x = [ self.initial_convolution(_x) for _x in x ]


        # Apply the main unet architecture:
        x = self.net_core(x)

        # Apply the final residual block to each plane:
        x = [ self.final_layer(_x) for _x in x ]
        x = [ self.bottleneck(_x) for _x in x ]


        # Might need to do some reshaping here
        x = torch.stack(x, 2)

        return x
