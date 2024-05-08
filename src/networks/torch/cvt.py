import torch
import sparseconvnet as scn

import numpy

from src.config.network import GrowthRate, DownSampling

class ConvolutionalAttention(torch.nn.Module):
    
    def __init__(self, nIn, params):
        super().__init__()

        self.num_heads = params.num_heads

        self.norm = torch.nn.LayerNorm(nIn)
        self.q, self.k, self.v = [
            torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels  = nIn,
                    out_channels = nIn,
                    kernel_size  = 3,
                    stride       = 1,
                    padding      = "same"
                ),
            ) for _ in range(3) ]

        self.attn = torch.nn.MultiheadAttention(nIn, self.num_heads, batch_first = True)

        self.mlp = torch.nn.Linear(nIn, nIn)


    def forward(self, x):
        

        q, k, v = self.q(x), self.k(x), self.v(x)


        input_shape = q.shape

        B = input_shape[0]
        c = input_shape[1]
        # h, w, d = input_shape[2], input_shape[3], input_shape[4]

        # This is the target shape before permuting:
        q_shape = (B, c, -1)

        q = torch.permute(torch.reshape(q, q_shape), (0,2,1))
        k = torch.permute(torch.reshape(k, q_shape), (0,2,1))
        v = torch.permute(torch.reshape(v, q_shape), (0,2,1))

        # Reshape for the number of heads:

        attn = self.attn(q, k, v, need_weights=False)[0]


        # Take the input, and shape it into flat tokens for the attention:
        token_x = torch.permute(torch.reshape(x, (B, c, -1)), (0,2,1))


        # Intermediate addition:
        inter_x = token_x + attn


        # Pass through the MLP
        output = inter_x + self.mlp(self.norm(inter_x))


        # Reshape the attention to match the input shape:
        output = torch.permute(output, (0,2,1))

        return torch.reshape(output, x.shape)


class ConvolutionalTokenEmbedding(torch.nn.Module):

    def __init__(self, nIn, nOut, params):
        super().__init__()

        self.conv = torch.nn.Conv2d(
            in_channels= nIn,
            out_channels= nOut,
            kernel_size = 7,
            stride = 2,
            padding = 3
        )


    def forward(self, x):

        x = self.conv(x)

        return x


class ConvolutionalTokenUpsampling(torch.nn.Module):

    def __init__(self, nIn, nOut, params):
        super().__init__()

        self.conv = torch.nn.ConvTranspose2d(
            in_channels= nIn,
            out_channels= nOut,
            kernel_size = 6,
            stride = 2,
            padding = 2
        )


    def forward(self, x):

        x = self.conv(x)

        return x



class BlockDown(torch.nn.Module):

    def __init__(self, nIn, nOut, params):
        super().__init__()

        self.cte = ConvolutionalTokenEmbedding(nIn, nOut, params)

        self.encoder_layers = torch.nn.Sequential()
        for _ in range(params.blocks_per_layer):
            self.encoder_layers.append(
                ConvolutionalAttention(nOut, params)
            )
            pass

    def forward(self, x):

        x = self.cte(x)

        x = self.encoder_layers(x)

        return x
    
class BlockUp(torch.nn.Module):

    def __init__(self, nIn, nOut, params):
        super().__init__()

        self.cte = ConvolutionalTokenUpsampling(nIn, nOut, params)

        self.encoder_layers = torch.nn.Sequential()
        for _ in range(params.blocks_per_layer):
            self.encoder_layers.append(
                ConvolutionalAttention(nOut, params)
            )
            pass

    def forward(self, x):

        x = self.cte(x)

        x = self.encoder_layers(x)

        return x
    


class DeepestBlock(torch.nn.Module):

    def __init__(self, nIn, params):
        torch.nn.Module.__init__(self)

        # We simply concatenate all tokens
        # Across all 3 planes

        # self.bottleneck = Block(
        #         inplanes   = 3*inplanes,
        #         outplanes  = n_filters_bottleneck,
        #         kernel     = [1,1],
        #         padding    = [0,0],
        #         params     = params)
        
        self.encoder_layers = torch.nn.Sequential()
        for _ in range(params.blocks_per_layer):
            self.encoder_layers.append(
                ConvolutionalAttention(3*nIn, params)
            )


        # self.blocks = BlockSeries(
        #     inplanes    = n_filters_bottleneck,
        #     n_blocks    = params.blocks_deepest_layer,
        #     params      = params)

        # self.unbottleneck = Block(
        #         inplanes   = n_filters_bottleneck,
        #         outplanes  = 3*inplanes,
        #         kernel     = [1,1],
        #         padding    = [0,0],
        #         params     = params)



    def forward(self, x):

        x = torch.cat(x, dim=1)
        x = self.encoder_layers(x)
        classification_head = x
        x = torch.chunk(x, chunks=3, dim=1)


        return x, classification_head, None # The none is a placeholder for vertex ID YOLO



class CoreLayer(torch.nn.Module):

    def __init__(self, depth, nIn, params):
        super().__init__()
        
        self.depth = depth
        self.vertex_depth = params.vertex.depth
        if depth == 0:
            # This is the deepest layer
            self.main_module = DeepestBlock(nIn, params)
            pass
        else:
            # We are at a mid-tier layer

            # Downsample spatial resolution:
            self.cte = ConvolutionalTokenEmbedding(nIn, 2*nIn, params)

            self.encoder_layers_down = torch.nn.Sequential()
            for _ in range(params.blocks_per_layer):
                self.encoder_layers_down.append(
                    ConvolutionalAttention(2*nIn, params)
                )

            # After downsampling, go into the next deepest layer
            self.main_module = CoreLayer(depth-1, 2*nIn, params)

            # After the next deepest layers, we upsample:
            self.ctu = ConvolutionalTokenUpsampling(2*nIn, nIn, params)

            # Apply some encoder layers:

            self.encoder_layers_up = torch.nn.Sequential()
            for _ in range(params.blocks_per_layer):
                self.encoder_layers_up.append(
                    ConvolutionalAttention(nIn, params)
                )

    def forward(self, x):

        # Take the input and apply the downward pass convolutions.  Save the residual
        # at the correct time.
        if self.depth != 0:

            residual = x

            x = tuple( self.cte(_x) for _x in x )

            # perform the downsampling operation:
            x = tuple( self.encoder_layers_down(_x) for _x in x )

        # Apply the main module:
        x, classification_head, vertex_head = self.main_module(x)

        # The vertex_head is None after the DEEPEST layer.  But, if we're returning it, do it here:


        if self.depth != 0:

            # perform the upsampling step:
            # perform the downsampling operation:
            x = tuple( self.ctu(_x) for _x in x )

            # Connect with the residual if necessary:
            # for i in range(len(x)):
            #     x[i] = self.connection(x[i], residual=residual[i])

            x = tuple( _x + _r for _x, _r in zip(x, residual))


            # Apply the convolutional steps:
            x = tuple( self.encoder_layers_up(_x) for _x in x )

        if self.depth == self.vertex_depth: vertex_head = x

        return x, classification_head, vertex_head

class CvT(torch.nn.Module):

    def __init__(self, params, image_size):

        super().__init__()


        self.patchify = torch.nn.Conv2d(
            in_channels  = 1,
            out_channels = params.n_initial_filters,
            kernel_size  = 7,
            stride       = 4,
            padding      = 2
        )

        nIn = params.n_initial_filters

        # self.network_layers = torch.nn.ModuleList()


        # for i_layer in range(params.depth):
        #     nOut = 2*nIn
        #     self.network_layers.append(Block(nIn, nOut, params))
        #     nIn = nOut

        # for i_layer in range(params.depth):
        #     nOut = int(0.5*nIn)
        #     self.network_layers.append(BlockUpsample(nIn, nOut, params))
        #     nIn = nOut

        self.net_core = CoreLayer(params.depth, nIn, params)

        # One more layer to get to the full resolution again:
        self.reshape = torch.nn.ConvTranspose2d(
                in_channels= nIn,
                out_channels= nIn,
                kernel_size = 4,
                stride  = 4,
                padding = 0
            )

        self.bottleneck = torch.nn.Conv2d(
            in_channels  = nIn,
            out_channels = 3,
            kernel_size  = 1,
            stride       = 1
        )



    def forward(self, x):

        return_dict = {
            "event_label" : None,
            "vertex"      : None,
        }




        x = torch.chunk(x, chunks=3, dim=1)

        # Apply the initial convolutions:
        x = tuple( self.patchify(_x) for _x in x )




        # for i, l in enumerate(self.network_layers):
        #     x = tuple( l(_x) for _x in x)

        x, classification, vertex = self.net_core(x)

        x = tuple( self.reshape(_x) for _x in x) 


        x = tuple( self.bottleneck(_x) for _x in x)

        
        # Pull off the classification token:
        # x = torch.mean(x, axis=(2,3,4))
        # print(x.shape)
        return_dict["segmentation"] = x

        return return_dict

