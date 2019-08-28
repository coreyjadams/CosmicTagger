import tensorflow as tf

class convolutional_block(tf.keras.models.Model):

    def __init__(self, *,
                 n_filters, 
                 kernel,
                 strides,
                 batch_norm,
                 data_format,
                 use_bias,
                 activation,
                 regularize):

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
        return  self.activation(x)


class convolutional_upsample(tf.keras.models.Model):

    def __init__(self, *,
        n_filters,
        kernel,
        strides,
        batch_norm,
        activation,
        data_format,
        use_bias,
        regularize):

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

    def __init__(self, *,
        n_filters,
        batch_norm,
        data_format,
        use_bias,
        regularize,
        bottleneck):

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
            print("Doing bottleneck")
            self.do_bottleneck = True
            self.bottleneck = convolutional_block(
                n_filters   = bottleneck,
                batch_norm  = batch_norm,
                data_format = data_format,
                use_bias    = use_bias,
                activation  = tf.nn.relu,
                kernel      = (1,1),
                strides     = (1,1),
                regularize  = regularize)

            n_filters = bottleneck

        self.convolution_1 = convolutional_block(
            n_filters   = n_filters,
            batch_norm  = batch_norm,
            data_format = data_format,
            use_bias    = use_bias,
            kernel      = (3,3),
            strides     = (1,1),
            activation  = tf.nn.relu,
            regularize  = regularize)

        self.convolution_2 = convolutional_block(
            n_filters   = n_filters_in,
            batch_norm  = batch_norm,
            data_format = data_format,
            use_bias    = use_bias,
            kernel      = (3,3),
            strides     = (1,1),
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


    def __init__(self, *, 
        in_filters,
        out_filters, 
        n_blocks, 
        residual, 
        data_format,
        batch_norm,
        use_bias,
        regularize):

        tf.keras.models.Model.__init__(self) 

        self.blocks = []
        if not residual:
            for i in range(n_blocks):
                self.blocks.append(
                    convolutional_block(
                        n_filters   = in_filters, 
                        data_format = data_format,
                        use_bias    = use_bias,
                        kernel      = (3,3),
                        strides     = (1,1),
                        activation  = tf.nn.relu,
                        batch_norm  = batch_norm,
                        regularize  = regularize
                    )
                 )


        else:
            # For the residual case, if in_filters != out_filters, we use a bottleneck:
            if in_filters != out_filters:
                self.blocks.append(
                    convolutional_block(
                        n_filters   = out_filters, 
                        data_format = data_format,
                        use_bias    = use_bias,
                        kernel      = (1,1),
                        strides     = (1,1),
                        activation  = tf.nn.relu,
                        batch_norm  = batch_norm,
                        regularize  = regularize
                    )
                )
            for i in range(n_blocks):
                self.blocks.append(
                    residual_block(
                        n_filters   = out_filters, 
                        data_format = data_format,
                        use_bias    = use_bias,
                        batch_norm  = batch_norm,
                        regularize  = regularize,
                        bottleneck  = -1,
                    )
                 )


    def call(self, x, training):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, training)

        return x


class DeepestBlock(tf.keras.models.Model):

    def __init__(self, in_filters, n_blocks,
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
            in_filters    = 3 * in_filters, 
            out_filters   = 3 * in_filters,
            n_blocks    = n_blocks, 
            residual    = residual,
            batch_norm  = batch_norm,
            data_format = data_format,
            use_bias    = use_bias,
            regularize  = regularize)



    def call(self, x, training):

        x = tf.concat(x, axis=self.channels_axis)
        x = self.blocks(x, training)

        x = tf.split(x, 3, self.channels_axis)
        return x


class NoConnection(tf.keras.models.Model):

    def __init__(self):
        tf.keras.models.Model.__init__(self)

    def call(self, x, residual, training):
        return x

class SumConnection(tf.keras.models.Model):

    def __init__(self):
        tf.keras.models.Model.__init__(self)

    def call(self, x, residual, training):
        return x + residual

class ConcatConnection(tf.keras.models.Model):
    
    def __init__(self, *,
            in_filters, 
            batch_norm,
            data_format,
            use_bias,
            activation,
            regularize):
        tf.keras.models.Model.__init__(self)


        if data_format == "channels_first":
            self.channels_axis = 1
        else:
            self.channels_axis = -1

        self.bottleneck = convolutional_block(
            n_filters = in_filters, 
            kernel = (1, 1),
            strides = (1,1),
            batch_norm = batch_norm,
            data_format = data_format,
            use_bias = use_bias,
            activation = tf.nn.relu,
            regularize = regularize,)

    def call(self, x, residual, training):
        x = tf.concat([x, residual] , axis=self.channels_axis)
        return self.bottleneck(x, training)


class MaxPooling(tf.keras.models.Model):

    def __init__(self,*, data_format):
        tf.keras.models.Model.__init__(self)
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=2, data_format=data_format) 
    def call(self, x, training):
        return self.pool(x)

class InterpolationUpsample(tf.keras.models.Model):

    def __init__(self, *,
        n_filters,
        data_format,
        batch_norm,
        use_bias,
        regularize):
        tf.keras.models.Model.__init__(self)


        self.up = tf.keras.layers.UpSampling2D(size=2, 
                    data_format=data_format, 
                    interpolation="bilinear")

        self.bottleneck = convolutional_block(
            n_filters   = n_filters,
            data_format = data_format,
            batch_norm  = batch_norm,
            strides     = (1,1),
            kernel      = (1,1),
            use_bias    = use_bias,
            activation  = tf.nn.relu,
            regularize  = regularize
            )

    def call(self, x, training):
        x = self.up(x)
        return self.bottleneck(x, training)

class UNetCore(tf.keras.models.Model):

    def __init__(self, *,
        blocks_deepest_layer, # How many blocks at the deepest layer
        depth,                # How many times to downsample and upsample
        nlayers,              # How many blocks to apply at this layer, if not deepest
        in_filters,           # How many filters are coming into this layer from above (or going up)
        out_filters,          # How many filters to pass to a deeper layer
        residual,             # Use residual blocks where possible
        data_format,          
        batch_norm,
        use_bias,
        regularize, 
        connections,
        upsampling,
        downsampling):


        if upsampling not in ["convolutional", "interpolation"]:
            raise Exception ("Must use either convolutional or interpolation upsampling")

        if downsampling not in ["convolutional", "max_pooling"]:
            raise Exception ("Must use either convolutional or max pooling downsampling")


        tf.keras.models.Model.__init__(self)
        

        self._depth_of_network  = depth
        self._number_of_layers = nlayers

        if depth == 0:
            self.main_module = DeepestBlock(in_filters, 
                n_blocks    = blocks_deepest_layer,
                residual    = residual, 
                data_format = data_format,
                batch_norm  = batch_norm,
                use_bias    = use_bias,
                regularize  = regularize)
        else:
            # Residual or convolutional blocks, applied in series:
            # The downsample pass will increase the number of filters if needed
            # There is no assumption that the number of "filters in" matches the planned
            # filter number for this layer
            self.down_blocks = BlockSeries(
                in_filters  = in_filters,
                out_filters   = out_filters,
                n_blocks    = nlayers, 
                residual    = residual, 
                data_format = data_format,
                batch_norm  = batch_norm,
                use_bias    = use_bias,
                regularize  = regularize)
    

            # Down sampling operation
            # This does not change the number of filters from above down-pass blocks
            if downsampling == "convolutional":
                self.downsample     = convolutional_block(
                    n_filters   = out_filters,
                    data_format = data_format,
                    batch_norm  = batch_norm,
                    strides     = (2,2),
                    kernel      = (2,2),
                    use_bias    = use_bias,
                    activation  = tf.nn.relu,
                    regularize  = regularize)
            else:
                self.downsample = MaxPooling(data_format=data_format)
            
            
            # Submodule: 
            self.main_module    = UNetCore(
                depth                    = depth-1, 
                blocks_deepest_layer     = blocks_deepest_layer,
                nlayers                  = nlayers, 
                in_filters               = out_filters, #passing in more filters
                out_filters              = 2*out_filters, # Double at the next layer too
                residual                 = residual, 
                data_format              = data_format,
                batch_norm               = batch_norm,
                use_bias                 = use_bias,
                regularize               = regularize,
                connections              = connections,
                upsampling               = upsampling,
                downsampling             = downsampling,)

            # Upsampling operation:
            # Upsampling will decrease the number of fitlers:
            if upsampling == "convolutional":
                self.upsample       = convolutional_upsample(
                    n_filters  = in_filters,
                    data_format = data_format,
                    batch_norm  = batch_norm,
                    use_bias    = use_bias,
                    regularize  = regularize,
                    kernel      = (2,2),
                    strides     = (2,2),
                    activation  = tf.nn.relu)
            else:
                self.upsample = InterpolationUpsample(
                    n_filters  = in_filters,
                    data_format = data_format,
                    batch_norm  = batch_norm,
                    use_bias    = use_bias,
                    regularize  = regularize)

            # Convolutional or residual blocks for the upsampling pass:

            # if the upsampling

            self.up_blocks = BlockSeries(
                in_filters  = in_filters,
                out_filters = in_filters, 
                n_blocks    = nlayers, 
                residual    = residual, 
                data_format = data_format,
                batch_norm  = batch_norm,
                use_bias    = use_bias,
                regularize  = regularize)



            # Residual connection operation:
            if connections == "sum":
                self.connection = SumConnection()
            elif connections == "concat":
                self.connection = ConcatConnection(
                    in_filters    = in_filters, 
                    batch_norm  = batch_norm,
                    data_format = data_format,
                    use_bias    = use_bias,
                    activation  = tf.nn.relu,
                    regularize  = regularize,)
            else:
                self.connection = NoConnection()


    def call(self, x, training):
        
        # print("depth ", self._depth_of_network, ", x[0] pre call ", x[0].shape)

        # Take the input and apply the downward pass convolutions.  Save the residual
        # at the correct time.
        if self._depth_of_network != 0:
            # Perform a series of convolutional or residual blocks:
            x = [ self.down_blocks(_x, training) for _x in x ]
            # print("depth ", self._depth_of_network, ", x[0] post resblocks shape ", x[0].shape)

            # Capture the residual right before downsampling:
            residual = x
            # print("depth ", self._depth_of_network, ", residual[0] shape ", residual[0].shape)

            # perform the downsampling operation:
            x = [ self.downsample(_x, training) for _x in x ]
            # print("depth ", self._depth_of_network, ", x[0] post downsample shape ", x[0].shape)

        # Apply the main module:

        # print("depth ", self._depth_of_network, ", x[0] pre main module shape ", x[0].shape)
        x = self.main_module(x, training)
        # print("depth ", self._depth_of_network, ", x[0] after main module shape ", x[0].shape)

        if self._depth_of_network != 0:

            # perform the upsampling step:
            # perform the downsampling operation:
            # print("depth ", self._depth_of_network, ", x[0] pre upsample shape ", x[0].shape)
            x = [ self.upsample(_x, training) for _x in x ]
            # print("depth ", self._depth_of_network, ", x[0] after upsample shape ", x[0].shape)


            x = [self.connection(residual[i], x[i], training) for i in range(len(x)) ]
            # print("depth ", self._depth_of_network, ", x[0] after connection shape ", x[0].shape)

            # Apply the convolutional steps:
            x = [ self.up_blocks(_x, training) for _x in x ]
            # print("depth ", self._depth_of_network, ", x[0] after res blocks shape ", x[0].shape)
            

        return x




class UResNet(tf.keras.models.Model):
    
    def __init__(self, *, n_initial_filters,
                    data_format,
                    batch_norm,
                    regularize,
                    depth,
                    residual,
                    use_bias,
                    blocks_final,
                    blocks_deepest_layer,
                    blocks_per_layer,
                    connections,
                    upsampling,
                    downsampling,):


        tf.keras.models.Model.__init__(self)
        

        if data_format == "channels_first":
            self.channels_axis = 1
        else:
            self.channels_axis = -1

        self.initial_convolution = convolutional_block(
            n_filters   = n_initial_filters,
            kernel      = (3,3),
            strides     = (1,1),
            batch_norm  = batch_norm,
            data_format = data_format,
            use_bias    = use_bias,
            activation  = tf.nn.relu,
            regularize  = regularize)

        n_filters = n_initial_filters
        # Next, build out the convolution steps:

        self.net_core = UNetCore(
            blocks_deepest_layer     = blocks_deepest_layer,
            depth                    = depth, 
            nlayers                  = blocks_per_layer, 
            in_filters               = n_initial_filters,
            out_filters              = n_initial_filters, 
            residual                 = residual,
            data_format              = data_format,
            batch_norm               = batch_norm,
            use_bias                 = use_bias,
            regularize               = regularize,
            connections              = connections,
            upsampling               = upsampling,
            downsampling             = downsampling,)


        # We need final output shaping too.  
        # Even with shared weights, keep this separate:
        self.final_blocks = False
        if blocks_final != 0:
            self.final_blocks = True
            self.final_layer = BlockSeries(
                in_filters  = n_filters, 
                out_filters = n_filters,
                n_blocks    = blocks_final,
                residual    = residual,
                data_format = data_format, 
                batch_norm  = batch_norm,
                use_bias    = use_bias,
                regularize  = regularize)


        self.bottleneck = convolutional_block(
            n_filters    = 3,
            kernel       = [1,1],
            strides      = [1,1],
            data_format  = data_format,
            batch_norm   = batch_norm,
            use_bias     = use_bias,
            activation   = tf.nn.relu,
            regularize   = regularize
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
        if self.final_blocks:
            x = [ self.final_layer(_x, training) for _x in x ]
        x = [ self.bottleneck(_x, training) for _x in x ]


        # Might need to do some reshaping here
        # x = tf.concat(x, self.channels_axis)

        return x