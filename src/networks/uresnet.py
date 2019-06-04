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

    def __init__(self, inplanes, outplanes, nplanes=1):
        nn.Module.__init__(self)

        padding = [0,1,1] if nplanes == 1 else [1,1,1]

        self.conv = nn.Conv3d(
            in_channels  = inplanes, 
            out_channels = outplanes, 
            kernel_size  = [nplanes, 3, 3],
            stride       = [1, 1, 1],
            padding      = padding,
            bias         = False)

        if FLAGS.BATCH_NORM:
            self.bn   = nn.BatchNorm3d(outplanes)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        if FLAGS.BATCH_NORM:
            out = self.bn(out)
        out = self.relu(out)
        return out


class ResidualBlock(nn.Module):

    def __init__(self, inplanes, outplanes, nplanes=1):
        nn.Module.__init__(self)

        padding = [0,1,1] if nplanes == 1 else [1,1,1]

        self.conv1 = nn.Conv3d(
            in_channels  = inplanes, 
            out_channels = outplanes, 
            kernel_size  = [nplanes, 3, 3],
            stride       = [1, 1, 1],
            padding      = padding,
            bias         = False)

        if FLAGS.BATCH_NORM:
            self.bn1 = nn.BatchNorm3d(outplanes)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(
            in_channels  = outplanes, 
            out_channels = outplanes, 
            kernel_size  = [nplanes, 3, 3],
            stride       = [1, 1, 1],
            padding      = padding,
            bias         = False)

        if FLAGS.BATCH_NORM:
            self.bn2 = nn.BatchNorm3d(outplanes)

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

    def __init__(self, inplanes, outplanes, nplanes=1):
        nn.Module.__init__(self)

        self.conv = nn.Conv3d(
            in_channels  = inplanes, 
            out_channels = outplanes, 
            kernel_size  = [nplanes,2,2],
            stride       = [1, 2, 2],
            padding      = [0, 0, 0],
            bias         = False)
        if FLAGS.BATCH_NORM:
            self.bn   = nn.BatchNorm3d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        if FLAGS.BATCH_NORM:
            out = self.bn(out)
        out = self.relu(out)
        return out


class ConvolutionUpsample(nn.Module):

    def __init__(self, inplanes, outplanes, nplanes=1):
        nn.Module.__init__(self)

        self.conv = nn.ConvTranspose3d(
            in_channels  = inplanes, 
            out_channels = outplanes, 
            kernel_size  = [nplanes, 2, 2],
            stride       = [1, 2, 2],
            padding      = [0, 0, 0],
            bias         = False)
        if FLAGS.BATCH_NORM:
            self.bn   = nn.BatchNorm3d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.conv(x)
        if FLAGS.BATCH_NORM:
            out = self.bn(out)
        out = self.relu(out)
        return out


class BlockSeries(torch.nn.Module):


    def __init__(self, inplanes, n_blocks, n_planes=1, residual=False):
        torch.nn.Module.__init__(self)

        if not residual:
            self.blocks = [ Convolution(inplanes, inplanes, n_planes) for i in range(n_blocks) ]
        else:
            self.blocks = [ ResidualBlock(inplanes, inplanes, n_planes) for i in range(n_blocks)]

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

        self.merger = nn.Conv3d(
            in_channels     = inplanes,
            out_channels    = FLAGS.NPLANES*inplanes,
            kernel_size     = [FLAGS.NPLANES,1,1],
            stride          = [1,1,1],
            padding         = [0, 0, 0],
            bias            = False)


        self.blocks = BlockSeries(FLAGS.NPLANES * inplanes, FLAGS.RES_BLOCKS_DEEPEST_LAYER, residual = residual)

        self.splitter = nn.ConvTranspose3d(
            in_channels     = FLAGS.NPLANES*inplanes,
            out_channels    = inplanes,
            kernel_size     = [FLAGS.NPLANES,1,1],
            stride          = [1,1,1],
            padding         = [0, 0, 0],
            bias            = False)




    def forward(self, x):
        
        x = self.merger(x)
        x = self.blocks(x)
        x = self.splitter(x)


        return x

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
            if FLAGS.SUMMED_CONNECTIONS:
                self.residual = torch.add
            else:
                self.residual = nn.Sequential(
                        torch.concat,
                        nn.Conv2d(2*inplanes, inplanes, kernel_size=1,stride=1,padding=1,bias=False)
                    )


    def forward(self, x):
        
        # Take the input and apply the downward pass convolutions.  Save the residual
        # at the correct time.
        if self.depth != 1:
            if FLAGS.CONNECT_PRE_RES_BLOCKS_DOWN:
                residual = x

            x = self.down_blocks(x)
            
            if not FLAGS.CONNECT_PRE_RES_BLOCKS_DOWN:
                residual = x

            # perform the downsampling operation:
            x = self.downsample(x)

        # Apply the main module:
        x = self.main_module(x)

        if self.depth != 1:

            # perform the upsampling step:
            # perform the downsampling operation:
            x = self.upsample(x)

            # Connect with the residual if necessary:
            if FLAGS.CONNECT_PRE_RES_BLOCKS_UP:
                x = self.residual(x, residual)


            # Apply the convolutional steps:
            x = self.up_blocks(x)
                
            if not FLAGS.CONNECT_PRE_RES_BLOCKS_UP:
                x = self.residual(x, residual)

        return x





class UResNet(torch.nn.Module):

    def __init__(self, shape):
        torch.nn.Module.__init__(self)


        

        self.initial_convolution = nn.Conv3d(
            in_channels  = 1,
            out_channels = FLAGS.N_INITIAL_FILTERS,
            kernel_size  = [1, 5, 5], 
            stride       = [1, 1, 1],
            padding      = [0, 2, 2], 
            bias         = False)

        n_filters = FLAGS.N_INITIAL_FILTERS
        # Next, build out the convolution steps:

        self.net_core = UNetCore(
            depth    = FLAGS.NETWORK_DEPTH, 
            nlayers  = FLAGS.RES_BLOCKS_PER_LAYER,
            inplanes = FLAGS.N_INITIAL_FILTERS,
            residual = True)

        # We need final output shaping too.  
        # Even with shared weights, keep this separate:

        self.final_layer = BlockSeries(
            FLAGS.N_INITIAL_FILTERS, 
            FLAGS.RES_BLOCKS_FINAL, 
            residual=True) 

        self.bottleneck = nn.Conv3d(
            FLAGS.N_INITIAL_FILTERS, 
            3, 
            kernel_size=1, 
            stride=1,
            padding=0,
            bias=False)

        # The rest of the final operations (reshape, softmax) are computed in the forward pass


        # Configure initialization:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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


        return x
