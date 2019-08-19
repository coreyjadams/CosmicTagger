import tensorflow as tf

class convolutional_block(tf.keras.models.Model):

    def __init__(self, 
                 n_filters, 
                 kernel     = (3,3),
                 strides    = (1,1),
                 batch_norm = True,
                 activation = tf.nn.relu,
                 data_format = 'channels_first',
                 use_bias   = False,
                 regularize = 0.0 ):

        tf.keras.models.Model.__init__(self)


        if data_format == "channels_first":
            self.channels_axis = 1
        else:
            self.channels_axis = -1

        self.convolution = tf.keras.layers.Conv2D(
            filters             = n_filters,
            kernel_size         = kernel,
            strides             = strides,
            padding             = 'same',
            activation          = None,
            use_bias            = use_bias,
            data_format         = data_format,
            kernel_regularizer  = tf.keras.regularizers.l2(l=regularize)
        )
        
        self.activation = activation
        
        if batch_norm:
            self._do_batch_norm = True
            self.batch_norm = tf.keras.layers.BatchNormalization(
                axis=self.channels_axis)
        else:
            self._do_batch_norm = False

    def call(self, inputs, training):

        x = self.convolution(inputs)
        if self._do_batch_norm:
            x = self.batch_norm(x)
        return self.activation(x)


class convolutional_upsample(tf.keras.models.Model):

    def __init__(self,
        n_filters   = None,
        kernel      = (2,2),
        strides     = (2,2),
        batch_norm  = True,
        activation  = tf.nn.relu,
        data_format = 'channels_first',
        use_bias    = False,
        regularize  = 0.0):

        tf.keras.models.Model.__init__(self)


        if data_format == "channels_first":
            self.channels_axis = 1
        else:
            self.channels_axis = -1

        self.convolution = tf.keras.layers.Conv2DTranspose(
            filters             = n_filters,
            kernel_size         = kernel,
            strides             = strides,
            padding             = 'same',
            activation          = None,
            use_bias            = use_bias,
            data_format         = data_format,
            kernel_regularizer  = tf.keras.regularizers.l2(l=regularize)
        )
        
        self.activation = activation
        

        if batch_norm:
            self._do_batch_norm = True
            self.batch_norm = tf.keras.layers.BatchNormalization(
                axis=self.channels_axis)
        else:
            self._do_batch_norm = False




    def call(self, inputs, training):

        x = self.convolution(inputs)
        if self._do_batch_norm:
            x = self.batch_norm(x)
        return self.activation(x)

class residual_block(tf.keras.models.Model):

    def __init__(self,
        n_filters,
        batch_norm  = True,
        data_format = "channels_first",
        use_bias    = False,
        regularize  = 0.0,
        bottleneck  = -1):

        tf.keras.models.Model.__init__(self)

        '''This is an all-in-one implementation of a residual block

        It can perform bottlenecking or not.
        
        Arguments:
            n_filters {int} -- number of filters
        Keyword Arguments:
            strides {tuple} -- [description] (default: {(1,1)})
            batch_norm {bool} -- [description] (default: {True})
            activation {[type]} -- [description] (default: {tf.nn.relu})
            name {str} -- [description] (default: {""})
            data_format {str} -- [description] (default: {"channels_first"})
            use_bias {bool} -- [description] (default: {False})
            regularize {number} -- [description] (default: {0.0})
            bottleneck {number} -- [description] (default: {64})
            reuse {bool} -- [description] (default: {False})
        '''



        n_filters_in = n_filters

        self.do_bottleneck = False
        if bottleneck != -1:
            self.do_bottleneck = True
            self.bottleneck = convolutional_block(
                n_filters   = bottleneck,
                batch_norm  = batch_norm,
                data_format = data_format,
                use_bias    = use_bias,
                regularize  = regularize)

            n_filters = bottleneck

        self.convolution_1 = convolutional_block(
            n_filters   = n_filters,
            batch_norm  = batch_norm,
            data_format = data_format,
            use_bias    = use_bias,
            regularize  = regularize)

        self.convolution_2 = convolutional_block(
            n_filters   = n_filters_in,
            batch_norm  = batch_norm,
            data_format = data_format,
            use_bias    = use_bias,
            activation  = tf.identity,
            regularize  = regularize)

    def call(self, inputs, training):

        x = inputs
        y = inputs

        if self.do_bottleneck:
            x = self.bottleneck(x, training)

        x = self.convolution_1(x, training)

        x = self.convolution_2(x, training)

        x = y + x

        return tf.nn.relu(x)


class BlockSeries(tf.keras.models.Model):


    def __init__(self, inplanes, n_blocks, 
        residual    = True, 
        data_format = "channels_first",
        batch_norm  = True,
        use_bias    = False,
        regularize  = 0.0):

        tf.keras.models.Model.__init__(self) 

        self.blocks = []
        if not residual:
            for i in range(n_blocks):
                self.blocks.append(
                    convolutional_block(inplanes, 
                        data_format = data_format,
                        use_bias    = use_bias,
                        batch_norm  = batch_norm,
                        regularize  = regularize
                    )
                 )


        else:
            for i in range(n_blocks):
                self.blocks.append(
                    residual_block(inplanes, 
                        data_format = data_format,
                        use_bias    = use_bias,
                        batch_norm  = batch_norm,
                        regularize  = regularize
                    )
                 )


    def call(self, x, training):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, training)

        return x


class DeepestBlock(tf.keras.models.Model):

    def __init__(self, inplanes, n_blocks,
        residual    = True, 
        data_format = "channels_first",
        batch_norm  = True,
        use_bias    = False,
        regularize  = 0.0):

        tf.keras.models.Model.__init__(self)
        


        if data_format == "channels_first":
            self.channels_axis = 1
        else:
            self.channels_axis = -1

        # The deepest block concats across planes, applies convolutions,
        # Then splits into planes again


        self.blocks = BlockSeries(
            inplanes   = 3 * inplanes, 
            n_blocks   = n_blocks, 
            residual   = residual,
            batch_norm = batch_norm,
            data_format = data_format,
            use_bias   = use_bias,
            regularize = regularize)



    def call(self, x, training):

        x = tf.concat(x, axis=self.channels_axis)

        x = self.blocks(x, training)

        x = tf.split(x, 3, self.channels_axis)

        return x

# class NoConnection(nn.Module):

#     def __init__(self):
#         nn.Module.__init__(self)

#     def forward(self, x, residual):
#         return x

# class SumConnection(nn.Module):

#     def __init__(self):
#         nn.Module.__init__(self)

#     def forward(self, x, residual):
#         return x + residual

# class ConcatConnection(nn.Module):
    
#     def __init__(self, inplanes):
#         nn.Module.__init__(self)

#         self.bottleneck = nn.Conv2d(
#             in_channels   = 2*inplanes, 
#             out_channels  = inplanes, 
#             kernel_size   = 1,
#             stride        = 1,
#             padding       = 0,
#             bias          = False)

#     def forward(self, x, residual):
#         x = torch.cat([x, residual], dim=1)
#         return self.bottleneck(x)


class UNetCore(tf.keras.models.Model):

    def __init__(self, 
        res_blocks_deepest_layer, 
        depth, nlayers, inplanes, 
        residual    = True, 
        data_format = "channels_first",
        batch_norm  = True,
        use_bias    = False,
        regularize  = 0.0):


        tf.keras.models.Model.__init__(self)
        

        self._depth_of_network  = depth
        self._number_of_layers = nlayers

        if depth == 1:
            self.main_module = DeepestBlock(inplanes, 
                n_blocks    = res_blocks_deepest_layer,
                residual    = residual, 
                data_format = data_format,
                batch_norm  = batch_norm,
                use_bias    = use_bias,
                regularize  = regularize)
        else:
            # Residual or convolutional blocks, applied in series:
            self.down_blocks = BlockSeries(inplanes, 
                n_blocks    = nlayers, 
                residual    = residual, 
                data_format = data_format,
                batch_norm  = batch_norm,
                use_bias    = use_bias,
                regularize  = regularize)
    
            n_filters_next_layer = inplanes * 2

            # Down sampling operation:
            self.downsample     = convolutional_block(
                n_filters   = n_filters_next_layer,
                data_format = data_format,
                batch_norm  = batch_norm,
                strides     = (2,2),
                use_bias    = use_bias,
                regularize  = regularize)
            
            
            # Submodule: 
            self.main_module    = UNetCore(
                depth       = depth-1, 
                res_blocks_deepest_layer = res_blocks_deepest_layer,
                nlayers     = nlayers, 
                inplanes    = n_filters_next_layer, 
                residual    = residual, 
                data_format = data_format,
                batch_norm  = batch_norm,
                use_bias    = use_bias,
                regularize  = regularize)

            # Upsampling operation:
            self.upsample       = convolutional_upsample(
                n_filters   = inplanes,
                data_format = data_format,
                batch_norm  = batch_norm,
                use_bias    = use_bias,
                regularize  = regularize)


            # Convolutional or residual blocks for the upsampling pass:
            self.up_blocks = BlockSeries(inplanes, nlayers, residual=residual)

            # # Residual connection operation:
            # if FLAGS.CONNECTIONS == "sum":
            #     self.connection = SumConnection()
            # elif FLAGS.CONNECTIONS == "concat":
            #     self.connection = ConcatConnection(inplanes)
            # else:
            #     self.connection = NoConnection()


    def call(self, x, training):
        
        # Take the input and apply the downward pass convolutions.  Save the residual
        # at the correct time.
        if self._depth_of_network != 1:
            residual = x
            x = [ self.down_blocks(_x, training) for _x in x ]
            # perform the downsampling operation:
            x = [ self.downsample(_x, training) for _x in x ]

        # Apply the main module:
        x = self.main_module(x, training)


        if self._depth_of_network != 1:

            # perform the upsampling step:
            # perform the downsampling operation:
            x = [ self.upsample(_x, training) for _x in x ]

            x = [residual[i] + x[i] for i in range(len(x)) ]

            # Apply the convolutional steps:
            x = [ self.up_blocks(_x, training) for _x in x ]
                
        return x



class UResNet(tf.keras.models.Model):
    '''
    Simple autoencoder forward model
    '''
    
    def __init__(self, n_initial_filters,
                    data_format,
                    batch_norm,
                    regularize,
                    depth,
                    residual,
                    use_bias,
                    res_blocks_final,
                    res_blocks_deepest_layer,
                    res_blocks_per_layer):


        tf.keras.models.Model.__init__(self)
        

        if data_format == "channels_first":
            self.channels_axis = 1
        else:
            self.channels_axis = -1

        self.initial_convolution = convolutional_block(
            n_filters = n_initial_filters,
            kernel      = (5,5),
            strides     = (1,1),
            batch_norm  = batch_norm,
            data_format = data_format,
            use_bias    = use_bias,
            regularize  = regularize)

        n_filters = n_initial_filters
        # Next, build out the convolution steps:

        self.net_core = UNetCore(
            res_blocks_deepest_layer = res_blocks_deepest_layer,
            depth    = depth, 
            nlayers  = res_blocks_per_layer,
            inplanes = n_filters,
            data_format = data_format,
            residual = residual)

        # We need final output shaping too.  
        # Even with shared weights, keep this separate:

        self.final_layer = BlockSeries(
            n_filters, 
            res_blocks_final, 
            data_format=data_format,
            residual=residual) 

        self.bottleneck = convolutional_block(
            n_filters    = 3,
            kernel       = [1,1],
            strides      = [1,1],
            data_format  = data_format,
            batch_norm   = batch_norm,
            use_bias     = use_bias,
        )



    def call(self, input_tensor, training):
        
        
        batch_size = input_tensor.get_shape()[0]

        print(batch_size)

        self.input_layer = tf.keras.layers.InputLayer()


        # Reshape this tensor into the right shape to apply this multiplane network.
        x = input_tensor
        x = tf.split(x, 3, self.channels_axis)



        # Apply the initial convolutions:
        x = [ self.initial_convolution(_x, training) for _x in x ]


        # Apply the main unet architecture:
        x = self.net_core(x, training)

        # Apply the final residual block to each plane:
        x = [ self.final_layer(_x, training) for _x in x ]
        x = [ self.bottleneck(_x, training) for _x in x ]


        # Might need to do some reshaping here
        # x = tf.concat(x, self.channels_axis)

        return x