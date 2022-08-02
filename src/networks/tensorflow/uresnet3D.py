import tensorflow as tf

from src.config.network import Connection, GrowthRate, DownSampling, UpSampling, Norm

class Block3D(tf.keras.models.Model):

    def __init__(self, *,
                 n_filters,
                 kernel  = [1,3,3],
                 strides = [1,1,1],
                 activation = tf.nn.leaky_relu,
                 params):

        tf.keras.models.Model.__init__(self)


        if params.data_format == "channels_first":
            self.channels_axis = 1
        else:
            self.channels_axis = -1

        self.convolution = tf.keras.layers.Conv3D(
            filters             = n_filters,
            kernel_size         = kernel,
            strides             = strides,
            padding             = 'same',
            activation          = None,
            use_bias            = params.bias,
            data_format         = params.data_format,
            kernel_regularizer  = tf.keras.regularizers.l2(l=params.weight_decay)
        )


        self.activation = activation

        if params.normalization == Norm.batch:
            self._do_normalization = True
            self.norm = tf.keras.layers.BatchNormalization(
                axis=self.channels_axis)
        elif params.normalization == Norm.layer:
            self._do_normalization = True
            self.norm = tf.keras.layers.LayerNormalization(
                axis=self.channels_axis)
        else:
            self._do_normalization = False

    def call(self, inputs, training):

        x = self.convolution(inputs)
        if self._do_normalization:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return  x

    def reg_loss(self):
        l = tf.reduce_sum(self.convolution.losses)

        return l


class ConvolutionUpsample3D(tf.keras.models.Model):

    def __init__(self, *,
        n_filters,
        kernel  = (1,2,2),
        strides = (1,2,2),
        activation = tf.nn.leaky_relu,
        params):

        tf.keras.models.Model.__init__(self)


        if params.data_format == "channels_first":
            self.channels_axis = 1
        else:
            self.channels_axis = -1

        self.convolution = tf.keras.layers.Conv3DTranspose(
            filters             = n_filters,
            kernel_size         = kernel,
            strides             = strides,
            padding             = 'same',
            activation          = None,
            use_bias            = params.bias,
            data_format         = params.data_format,
            kernel_regularizer  = tf.keras.regularizers.l2(l=params.weight_decay)
        )

        self.activation = activation


        if params.normalization == Norm.batch:
            self._do_normalization = True
            self.norm = tf.keras.layers.BatchNormalization(
                axis=self.channels_axis)
        elif params.normalization == Norm.layer:
            self._do_normalization = True
            self.norm = tf.keras.layers.LayerNormalization(
                axis=self.channels_axis)
        else:
            self._do_normalization = False





    def call(self, inputs, training):

        x = self.convolution(inputs)
        if self._do_normalization:
            x = self.norm(x)
        return self.activation(x)

    def reg_loss(self):
        l = tf.reduce_sum(self.convolution.losses)

        return l

class ResidualBlock3D(tf.keras.models.Model):

    def __init__(self, *, kernel,
        n_filters, params):

        tf.keras.models.Model.__init__(self)

        '''This is an all-in-one implementation of a residual block

        It can perform bottlenecking or not.

        Arguments:
            n_filters {int} -- number of filters
        Keyword Arguments:
            strides {tuple} -- [description] (default: {(1,1)})
            batch_norm {bool} -- [description] (default: {True})
            activation {[type]} -- [description] (default: {tf.nn.leaky_relu})
            name {str} -- [description] (default: {""})
            data_format {str} -- [description] (default: {"channels_first"})
            use_bias {bool} -- [description] (default: {False})
            regularize {number} -- [description] (default: {0.0})
            bottleneck {number} -- [description] (default: {64})
            reuse {bool} -- [description] (default: {False})
        '''



        n_filters_in = n_filters

        self.convolution_1 = Block3D(
            n_filters   = n_filters,
            kernel      = kernel,
            params      = params)

        self.convolution_2 = Block3D(
            n_filters   = n_filters_in,
            activation  = tf.identity,
            kernel      = kernel,
            params      = params)

    def call(self, inputs, training):

        x = inputs
        y = inputs

        x = self.convolution_1(x, training)

        x = self.convolution_2(x, training)

        x = y + x

        return tf.nn.leaky_relu(x)

    def reg_loss(self):
        l = self.convolution_1.reg_loss() + self.convolution_2.reg_loss()

        return l


class BlockSeries3D(tf.keras.models.Model):


    def __init__(self, *,
        out_filters,
        n_blocks,
        kernel = [1,3,3],
        params):

        tf.keras.models.Model.__init__(self)

        self.blocks = []
        if not params.residual:
            for i in range(n_blocks):
                self.blocks.append(
                    Block3D(
                        n_filters   = out_filters,
                        kernel      = kernel,
                        params      = params
                    )
                 )


        else:
            for i in range(n_blocks):
                self.blocks.append(
                    ResidualBlock3D(
                        n_filters   = out_filters,
                        kernel      = kernel,
                        params      = params
                    )
                 )


    def call(self, x, training):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, training)

        return x

    def reg_loss(self):
        l = tf.reduce_sum([b.reg_loss() for b in self.blocks])

        return l

class DeepestBlock3D(tf.keras.models.Model):

    def __init__(self, in_filters, params):

        tf.keras.models.Model.__init__(self)


        if params.data_format == "channels_first":
            self.channels_axis = 1
        else:
            self.channels_axis = -1

        # The deepest block concats across planes, applies convolutions,
        # Then splits into planes again
        n_filters_bottleneck = params.bottleneck_deepest
        self.bottleneck = Block3D(
                 n_filters  = n_filters_bottleneck,
                 kernel     = [3,1,1],
                 strides    = [1,1,1],
                 params     = params)

        self.blocks = BlockSeries3D(
            out_filters = n_filters_bottleneck,
            kernel      = [1,
                           params.filter_size_deepest,
                           params.filter_size_deepest],
            n_blocks    = params.blocks_deepest_layer,
            params      = params)

        self.unbottleneck = Block3D(
                n_filters  = in_filters,
                kernel     = [3,1,1],
                strides    = [1,1,1],
                params     = params)

        # self.blocks = BlockSeries3D(
        #     out_filters = n_filters,
        #     n_blocks    = params.blocks_deepest_layer,
        #     kernel      = [3,3,3],
        #     params      = params)



    def call(self, x, training):
        x = self.bottleneck(x)
        x = self.blocks(x, training)
        x = self.unbottleneck(x)

        return x


    def reg_loss(self):

        l = self.bottleneck.reg_loss() + self.blocks.reg_loss() + self.unbottleneck.reg_loss()
        return l

class NoConnection3D(tf.keras.models.Model):

    def __init__(self):
        tf.keras.models.Model.__init__(self)

    def call(self, x, residual, training):
        return x

    def reg_loss(self): return 0.0

class SumConnection3D(tf.keras.models.Model):

    def __init__(self):
        tf.keras.models.Model.__init__(self)

    def call(self, x, residual, training):
        return x + residual

    def reg_loss(self): return 0.0

class ConcatConnection3D(tf.keras.models.Model):

    def __init__(self, *,
            in_filters,
            params):
        tf.keras.models.Model.__init__(self)


        if params.data_format == "channels_first":
            self.channels_axis = 1
        else:
            self.channels_axis = -1

        self.bottleneck = Block3D(
            n_filters   = in_filters,
            kernel      = (1,1,1),
            strides     = (1,1,1),
            params      = params)

    def call(self, x, residual, training):
        x = tf.concat([x, residual] , axis=self.channels_axis)
        x = self.bottleneck(x, training)
        return x

    def reg_loss(self):
        return self.bottleneck.reg_loss()

class MaxPooling3D(tf.keras.models.Model):

    def __init__(self,*,
            n_filters,
            params):
        tf.keras.models.Model.__init__(self)


        self.pool = tf.keras.layers.MaxPooling3D(pool_size=[1,2,2], data_format=params.data_format)

        self.bottleneck = Block3D(
            n_filters   = n_filters,
            kernel      = (1,1,1),
            strides     = (1,1,1),
            params      = params)

    def call(self, x, training):
        x = self.pool(x)

        return self.bottleneck(x, training)

    def reg_loss(self): return 0.0


class InterpolationUpsample3D(tf.keras.models.Model):

    def __init__(self, *,
        n_filters,
        params):
        tf.keras.models.Model.__init__(self)


        self.up = tf.keras.layers.UpSampling3D(size=[1,2,2],
                    data_format=params.data_format)

        self.bottleneck = Block3D(
            n_filters   = n_filters,
            strides     = (1,1,1),
            kernel      = (1,1,1),
            params      = params,
            )

    def call(self, x, training):
        x = self.up(x)
        return self.bottleneck(x, training)


    def reg_loss(self):
        return self.bottleneck.reg_loss()


class UNetCore3D(tf.keras.models.Model):

    def __init__(self, *,
        depth,
        in_filters,        # How many filters are coming into this layer from above (or going up)
        out_filters,       # How many filters to pass to a deeper layer
        params ):        # Wha


        # if params.upsampling not in ["convolutional", "interpolation"]:
        #     raise Exception ("Must use either convolutional or interpolation upsampling")

        # if params.downsampling not in ["convolutional", "max_pooling"]:
        #     raise Exception ("Must use either convolutional or max pooling downsampling")

        # if params.connections not in ['sum', 'concat', 'none']:
        #     if params.connections != None:
        #         raise Exception("Don't know what to do with connection type ", params.connections)


        tf.keras.models.Model.__init__(self)


        self._depth_of_network  = depth

        if depth == 0:
            self.main_module = DeepestBlock3D(in_filters,
                params      = params)
        else:
            # Residual or convolutional blocks, applied in series:
            # Doesn't change the number of filters
            self.down_blocks = BlockSeries3D(
                out_filters = in_filters,
                n_blocks    = params.blocks_per_layer,
                kernel      = [1,3,3],
                params      = params)

            # Down sampling operation
            # This does change the number of filters from above down-pass blocks
            if params.downsampling == DownSampling.convolutional:
                self.downsample = Block3D(
                    n_filters   = out_filters,
                    strides     = (1,2,2),
                    kernel      = (1,2,2),
                    params      = params)
            else:
                self.downsample = MaxPooling3D(
                    n_filters   = out_filters,
                    params      = params)


            if params.growth_rate == GrowthRate.multiplicative:
                n_filters_next = 2 * out_filters
            else:
                n_filters_next = out_filters + params.n_initial_filters

            # Submodule:
            self.main_module    = UNetCore3D(
                depth           = depth-1,
                in_filters      = out_filters, #passing in more filters
                out_filters     = n_filters_next, # Double at the next layer too
                params          = params
            )

            # Upsampling operation:
            # Upsampling will decrease the number of fitlers:

            if params.upsampling == UpSampling.convolutional:
                self.upsample = ConvolutionUpsample3D(
                    n_filters   = in_filters,
                    kernel      = (1,2,2),
                    strides     = (1,2,2),
                    params      = params)
            else:
                # raise Exception("This ought to be unreachable!")
                self.upsample = InterpolationUpsample3D(
                    n_filters   = in_filters,
                    params      = params)

            # Convolutional or residual blocks for the upsampling pass:

            # if the upsampling

            self.up_blocks = BlockSeries3D(
                out_filters = in_filters,
                n_blocks    = params.blocks_per_layer,
                kernel      = [1,3,3],
                params      = params)



            # Residual connection operation:
            if params.connections == Connection.sum:
                self.connection = SumConnection3D()
            elif params.connections == Connection.concat:
                # Concat applies a concat + bottleneck
                self.connection = ConcatConnection3D(
                    in_filters  = in_filters,
                    params      = params)
            else:
                self.connection = NoConnection3D()


    def call(self, x, training):

        # print("depth ", self._depth_of_network, ", x[0] pre call ", x[0].shape)

        # Take the input and apply the downward pass convolutions.  Save the residual
        # at the correct time.
        if self._depth_of_network != 0:
            # Perform a series of convolutional or residual blocks:
            x = self.down_blocks(x, training)
            # print("depth ", self._depth_of_network, ", x[0] post resblocks shape ", x[0].shape)

            # Capture the residual right before downsampling:
            residual = x
            # print("depth ", self._depth_of_network, ", residual[0] shape ", residual[0].shape)

            # perform the downsampling operation:
            x = self.downsample(x, training)
            # print("depth ", self._depth_of_network, ", x[0] post downsample shape ", x[0].shape)

        # Apply the main module:

        # print("depth ", self._depth_of_network, ", x[0] pre main module shape ", x[0].shape)
        x = self.main_module(x, training)
        # print("depth ", self._depth_of_network, ", x[0] after main module shape ", x[0].shape)

        if self._depth_of_network != 0:

            # perform the upsampling step:
            # perform the downsampling operation:
            # print("depth ", self._depth_of_network, ", x[0] pre upsample shape ", x[0].shape)
            x = self.upsample(x, training)
            # print("depth ", self._depth_of_network, ", x[0] after upsample shape ", x[0].shape)


            # Apply the convolutional steps:
            x = self.up_blocks(x, training)
            # print("depth ", self._depth_of_network, ", x[0] after res blocks shape ", x[0].shape)

            x = self.connection(residual, x, training)
            # print("depth ", self._depth_of_network, ", x[0] after connection shape ", x[0].shape)

        return x



    def reg_loss(self):

        l = self.main_module.reg_loss()

        if self._depth_of_network != 0:
            l += self.down_blocks.reg_loss()
            l += self.downsample.reg_loss()
            l += self.upsample.reg_loss()
            l += self.up_blocks.reg_loss()
            l += self.connection.reg_loss()

        return l

class UResNet3D(tf.keras.models.Model):

    def __init__(self, params):

        tf.keras.models.Model.__init__(self)


        if params.data_format == "channels_first":
            self.channels_axis = 1
        else:
            self.channels_axis = -1

        self.initial_convolution = Block3D(
            n_filters   = params.n_initial_filters,
            kernel      = [1,5,5],
            activation  = tf.nn.leaky_relu,
            params      = params)

        n_filters = params.n_initial_filters
        # Next, build out the convolution steps:

        n_filters_next = 2 * params.n_initial_filters

        self.net_core = UNetCore3D(
            depth                    = params.depth,
            in_filters               = params.n_initial_filters,
            out_filters              = n_filters_next,
            params                   = params)


        # We need final output shaping too.
        self.final_blocks = False
        if params.blocks_final != 0:
            self.final_blocks = True
            self.final_layer = BlockSeries3D(
                out_filters = n_filters,
                n_blocks    = params.blocks_final,
                kernel      = [1,3,3],
                params      = params)



        self.bottleneck = tf.keras.layers.Conv3D(
            filters             = 3,
            kernel_size         = [1,1,1],
            strides             = [1,1,1],
            activation          = None,
            use_bias            = params.bias,
            data_format         = params.data_format,
            kernel_regularizer  = tf.keras.regularizers.l2(l=params.weight_decay)
        )
        self.weight_decay = params.weight_decay

    @tf.function
    def reg_loss(self):
        l = tf.reduce_sum(self.initial_convolution.losses)
        l += self.net_core.reg_loss()
        if self.final_blocks:
            l += self.final_layer.reg_loss()
        l += tf.reduce_sum(self.bottleneck.losses)

        return self.weight_decay * tf.sqrt(l)

    def call(self, input_tensor, training):


        batch_size = input_tensor.get_shape()[0]


        # Reshape this tensor into the right shape to apply this multiplane network.
        x = input_tensor

        # If data_format is "channels_first", we expect the shape to be:
        # [B, 3, H, W] and we need it to be:
        # [B, 1, 3, H, W]

        # If dataformat is 'channels_last', we expect the shape to be:
        # [B, H, W, 3] and we need it to be:
        # [B, 3, H, W, 1]



        # print(x.get_shape())

        if self.channels_axis == 1:
            x = tf.expand_dims(x, 1)
        else:
            # print(x.get_shape())
            x = tf.expand_dims(x, -1)
            # print(x.get_shape())
            # Shape here is [B, H, W, 3, 1]
            x = tf.transpose(a=x, perm=[0,3,1,2,4])
            # print(x.get_shape())


        # print(x.get_shape())

        # Apply the initial convolutions:
        x = self.initial_convolution(x, training)


        # Apply the main unet architecture:
        x = self.net_core(x, training)

        # Apply the final residual block to each plane:
        if self.final_blocks:
            x = self.final_layer(x, training)

        # x = [ tf.concat([x[i], split_input[i]], axis=self.channels_axis) for i in range(3)]
        x = self.bottleneck(x)


        # For the 3D conv case, we are still using 2D loss functions.
        # So, we need to do some splitting here:
        # print(x.get_shape())

        # If data_format is "channels_first", we expect the shape to be:
        # [B, 3, 3, H, W] and we need it to be:
        # 3 tensors of [B, 3, 1, H, W] which gets reshaped to [B, 3, H, W]

        # If dataformat is 'channels_last', we expect the shape to be:
        # [B, 3, H, W, 3] and we need it to be:
        # 3 tensors of [B, 1, H, W, 3] which gets reshaped to [B, H, W, 3]

        if self.channels_axis == 1:
            x = tf.split(x, 3, axis=2)
            x = [ tf.squeeze(_x, axis=2) for _x in x]
        elif self.channels_axis == -1:
            x = tf.split(x, 3, axis=1)
            x = [ tf.squeeze(_x, axis=1) for _x in x]

        # for _x in x: print(_x.get_shape())
        # Might need to do some reshaping here
        # x = tf.concat(x, self.channels_axis)

        return x
