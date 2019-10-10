import tensorflow as tf

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


class Block(tf.keras.models.Model):

    def __init__(self, *,
                 n_filters,
                 kernel  = [3,3],
                 strides = [1,1],
                 activation = tf.nn.relu,
                 params):

        tf.keras.models.Model.__init__(self)


        if params.data_format == "channels_first":
            self.channels_axis = 1
        else:
            self.channels_axis = -1

        self.convolution = tf.keras.layers.Conv2D(
            filters             = n_filters,
            kernel_size         = kernel,
            strides             = strides,
            padding             = 'same',
            activation          = None,
            use_bias            = params.use_bias,
            data_format         = params.data_format,
            kernel_regularizer  = tf.keras.regularizers.l2(l=params.regularize)
        )


        self.activation = activation

        if params.batch_norm:
            self._do_batch_norm = True
            self.batch_norm = tf.keras.layers.BatchNormalization(
                axis=self.channels_axis)
        else:
            self._do_batch_norm = False

    def call(self, inputs, training):

        x = self.convolution(inputs)
        if self._do_batch_norm:
            x = self.batch_norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return  x


class ConvolutionUpsample(tf.keras.models.Model):

    def __init__(self, *,
        n_filters,
        kernel  = (2,2),
        strides = (2,2),
        activation = tf.nn.relu,
        params):

        tf.keras.models.Model.__init__(self)


        if params.data_format == "channels_first":
            self.channels_axis = 1
        else:
            self.channels_axis = -1

        self.convolution = tf.keras.layers.Conv2DTranspose(
            filters             = n_filters,
            kernel_size         = kernel,
            strides             = strides,
            padding             = 'same',
            activation          = None,
            use_bias            = params.use_bias,
            data_format         = params.data_format,
            kernel_regularizer  = tf.keras.regularizers.l2(l=params.regularize)
        )

        self.activation = activation


        if params.batch_norm:
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

class ResidualBlock(tf.keras.models.Model):

    def __init__(self, *,
        n_filters, params):

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

        self.convolution_1 = Block(
            n_filters   = n_filters,
            params      = params)

        self.convolution_2 = Block(
            n_filters   = n_filters_in,
            activation  = tf.identity,
            params      = params)

    def call(self, inputs, training):

        x = inputs
        y = inputs

        x = self.convolution_1(x, training)

        x = self.convolution_2(x, training)

        x = y + x

        return tf.nn.relu(x)


class BlockSeries(tf.keras.models.Model):


    def __init__(self, *,
        out_filters,
        n_blocks,
        params):

        tf.keras.models.Model.__init__(self)

        self.blocks = []
        if not params.residual:
            for i in range(n_blocks):
                self.blocks.append(
                    Block(
                        n_filters   = out_filters,
                        params      = params
                    )
                 )


        else:
            for i in range(n_blocks):
                self.blocks.append(
                    ResidualBlock(
                        n_filters   = out_filters,
                        params      = params
                    )
                 )


    def call(self, x, training):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, training)

        return x


class DeepestBlock(tf.keras.models.Model):

    def __init__(self, in_filters, params):

        tf.keras.models.Model.__init__(self)



        if params.data_format == "channels_first":
            self.channels_axis = 1
        else:
            self.channels_axis = -1

        # The deepest block concats across planes, applies convolutions,
        # Then splits into planes again


        self.blocks = BlockSeries(
            out_filters = 3 * in_filters,
            n_blocks    = params.blocks_deepest_layer,
            params      = params)



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
            params):
        tf.keras.models.Model.__init__(self)


        if params.data_format == "channels_first":
            self.channels_axis = 1
        else:
            self.channels_axis = -1

        self.bottleneck = Block(
            n_filters   = in_filters,
            kernel      = (1, 1),
            strides     = (1,1),
            params      = params)

    def call(self, x, residual, training):
        x = tf.concat([x, residual] , axis=self.channels_axis)
        x = self.bottleneck(x, training)
        return x


class MaxPooling(tf.keras.models.Model):

    def __init__(self,*,
            n_filters,
            params):
        tf.keras.models.Model.__init__(self)


        self.pool = tf.keras.layers.MaxPooling2D(pool_size=2, data_format=params.data_format)

        self.bottleneck = Block(
            n_filters   = n_filters,
            kernel      = (1,1),
            strides     = (1,1),
            params      = params)

    def call(self, x, training):
        x = self.pool(x)

        return self.bottleneck(x, training)

class InterpolationUpsample(tf.keras.models.Model):

    def __init__(self, *,
        n_filters,
        params):
        tf.keras.models.Model.__init__(self)


        self.up = tf.keras.layers.UpSampling2D(size=2,
                    data_format=params.data_format,
                    interpolation="bilinear")

        self.bottleneck = Block(
            n_filters   = n_filters,
            strides     = (1,1),
            kernel      = (1,1),
            params      = params,
            )

    def call(self, x, training):
        x = self.up(x)
        return self.bottleneck(x, training)

class UNetCore(tf.keras.models.Model):

    def __init__(self, *,
        depth,
        in_filters,        # How many filters are coming into this layer from above (or going up)
        out_filters,       # How many filters to pass to a deeper layer
        params ):        # Wha


        if params.upsampling not in ["convolutional", "interpolation"]:
            raise Exception ("Must use either convolutional or interpolation upsampling")

        if params.downsampling not in ["convolutional", "max_pooling"]:
            raise Exception ("Must use either convolutional or max pooling downsampling")

        if params.connections not in ['sum', 'concat', 'none']:
            if params.connections != None:
                raise Exception("Don't know what to do with connection type ", params.connections)


        tf.keras.models.Model.__init__(self)


        self._depth_of_network  = depth

        if depth == 0:
            self.main_module = DeepestBlock(in_filters,
                params      = params)
        else:
            # Residual or convolutional blocks, applied in series:
            # Doesn't change the number of filters
            self.down_blocks = BlockSeries(
                out_filters = in_filters,
                n_blocks    = params.blocks_per_layer,
                params      = params)

            # Down sampling operation
            # This does change the number of filters from above down-pass blocks
            if params.downsampling == "convolutional":
                self.downsample = Block(
                    n_filters   = out_filters,
                    strides     = (2,2),
                    kernel      = (2,2),
                    params      = params)
            else:
                self.downsample = MaxPooling(
                    n_filters   = out_filters,
                    params      = params)


            # Submodule:
            self.main_module    = UNetCore(
                depth           = depth-1,
                in_filters      = out_filters, #passing in more filters
                out_filters     = 2*out_filters, # Double at the next layer too
                params          = params
            )

            # Upsampling operation:
            # Upsampling will decrease the number of fitlers:

            if params.upsampling == "convolutional":
                self.upsample = ConvolutionUpsample(
                    n_filters   = in_filters,
                    kernel      = (2,2),
                    strides     = (2,2),
                    params      = params)
            else:
                self.upsample = InterpolationUpsample(
                    n_filters   = in_filters,
                    params      = params)

            # Convolutional or residual blocks for the upsampling pass:

            # if the upsampling

            self.up_blocks = BlockSeries(
                out_filters = in_filters,
                n_blocks    = params.blocks_per_layer,
                params      = params)



            # Residual connection operation:
            if params.connections == "sum":
                self.connection = SumConnection()
            elif params.connections == "concat":
                # Concat applies a concat + bottleneck
                self.connection = ConcatConnection(
                    in_filters  = in_filters,
                    params      = params)
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


            # Apply the convolutional steps:
            x = [ self.up_blocks(_x, training) for _x in x ]
            # print("depth ", self._depth_of_network, ", x[0] after res blocks shape ", x[0].shape)

            x = [self.connection(residual[i], x[i], training) for i in range(len(x)) ]
            # print("depth ", self._depth_of_network, ", x[0] after connection shape ", x[0].shape)

        return x




class UResNet(tf.keras.models.Model):

    def __init__(self, *, n_initial_filters,
                    data_format,          # Channels first or channels last
                    batch_norm,           # Use Batch norm?
                    regularize,           # Apply weight regularization?
                    depth,                # How many times to downsample and upsample
                    residual,             # Use residual blocks where possible
                    use_bias,             # Use Bias layers?
                    blocks_final,         # How many blocks just before bottleneck?
                    blocks_deepest_layer, # How many blocks at the deepest layer
                    blocks_per_layer,     # How many blocks to apply at this layer, if not deepest
                    connections,          # What type of connection?
                    upsampling,           # What type of upsampling?
                    downsampling          # What type of downsampling?
                ):

        tf.keras.models.Model.__init__(self)


        params = objectview({
            'n_initial_filters'     : n_initial_filters,
            'data_format'           : data_format,
            'batch_norm'            : batch_norm,
            'use_bias'              : use_bias,
            'residual'              : residual,
            'regularize'            : regularize,
            'depth'                 : depth,
            'blocks_final'          : blocks_final,
            'blocks_per_layer'      : blocks_per_layer,
            'blocks_deepest_layer'  : blocks_deepest_layer,
            'connections'           : connections,
            'upsampling'            : upsampling,
            'downsampling'          : downsampling,
            })

        if data_format == "channels_first":
            self.channels_axis = 1
        else:
            self.channels_axis = -1

        self.initial_convolution = Block(
            n_filters   = n_initial_filters,
            kernel      = [7,7],
            activation  = tf.nn.relu,
            params      = params)

        n_filters = n_initial_filters
        # Next, build out the convolution steps:

        self.net_core = UNetCore(
            depth                    = depth,
            in_filters               = n_initial_filters,
            out_filters              = 2*n_initial_filters,
            params                   = params)


        # We need final output shaping too.
        self.final_blocks = False
        if blocks_final != 0:
            self.final_blocks = True
            self.final_layer = BlockSeries(
                out_filters = n_filters,
                n_blocks    = blocks_final,
                params      = params)


        self.bottleneck = Block(
            n_filters    = 3,
            kernel       = [1,1],
            strides      = [1,1],
            params       = params,
            activation   = None,

        )



    def call(self, input_tensor, training):


        batch_size = input_tensor.get_shape()[0]


        # Reshape this tensor into the right shape to apply this multiplane network.
        x = input_tensor
        x = tf.split(x, 3, self.channels_axis)
        split_input = x



        # Apply the initial convolutions:
        x = [ self.initial_convolution(_x, training) for _x in x ]


        # Apply the main unet architecture:
        x = self.net_core(x, training)

        # Apply the final residual block to each plane:
        if self.final_blocks:
            x = [ self.final_layer(_x, training) for _x in x ]

        # x = [ tf.concat([x[i], split_input[i]], axis=self.channels_axis) for i in range(3)]
        x = [ self.bottleneck(_x, training) for _x in x ]


        # Might need to do some reshaping here
        # x = tf.concat(x, self.channels_axis)

        return x
