import sys

import tensorflow as tf
from ..utils import flags
FLAGS = flags.FLAGS()

def convolutional_block(input_tensor,
                        is_training,
                        n_filters  = None,
                        kernel     = (3,3),
                        strides    = (1,1),
                        batch_norm = True,
                        activation = tf.nn.relu,
                        data_format = 'channels_first',
                        name       = "",
                        use_bias   = False,
                        regularize = 0.0,
                        reuse      = False):


    with tf.variable_scope(name):
        x = input_tensor

        if n_filters is None:
            if data_format == "channels_last":
                n_filters = x.shape[-1]
            else:
                n_filters = x.shape[1]


        x = tf.layers.conv2d(x, n_filters,
                             kernel_size=kernel,
                             strides=strides,
                             padding='same',
                             activation=None,
                             use_bias=use_bias,
                             trainable=is_training,
                             data_format=data_format,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=regularize),
                             name="Convolution",
                             reuse=reuse)

        # Apply batchnorm, if needed
        if batch_norm:
            with tf.variable_scope("batch_norm") as scope:
                # updates_collections=None is important here
                # it forces batchnorm parameters to be saved immediately,
                # rather than have to be saved into snapshots manually.
                x = tf.contrib.layers.batch_norm(x,
                                                 updates_collections=None,
                                                 decay=0.9,
                                                 is_training=is_training,
                                                 trainable=is_training,
                                                 scope=scope,
                                                 # name="BatchNorm",
                                                 reuse=reuse)

        # Apply the activation:
        if activation is not None:
            x = activation(x)

    return x



def convolutional_upsample(input_tensor,
                        is_training,
                        n_filters  = None,
                        kernel     = (2,2),
                        strides    = (2,2),
                        batch_norm = True,
                        activation = tf.nn.relu,
                        data_format = 'channels_first',
                        name       = "",
                        use_bias   = False,
                        regularize = 0.0,
                        reuse      = False):


    with tf.variable_scope(name):
        x = input_tensor

        if n_filters is None:
            if data_format == "channels_last":
                n_filters = x.shape[-1]
            else:
                n_filters = x.shape[1]


        x = tf.layers.conv2d_transpose(x, n_filters,
                                       kernel_size=kernel,
                                       strides=strides,
                                       padding='same',
                                       activation=None,
                                       data_format=data_format,
                                       use_bias=use_bias,
                                       trainable=is_training,
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=regularize),
                                       name="Convolution",
                                       reuse=reuse)

        # Apply batchnorm, if needed
        if batch_norm:
            with tf.variable_scope("batch_norm") as scope:
                # updates_collections=None is important here
                # it forces batchnorm parameters to be saved immediately,
                # rather than have to be saved into snapshots manually.
                x = tf.contrib.layers.batch_norm(x,
                                                 updates_collections=None,
                                                 decay=0.9,
                                                 is_training=is_training,
                                                 trainable=is_training,
                                                 scope=scope,
                                                 # name="BatchNorm",
                                                 reuse=reuse)

        # Apply the activation:
        if activation is not None:
            x = activation(x)

    return x


def residual_block(input_tensor,
                   is_training,
                   strides    = (1,1),
                   batch_norm = True,
                   activation = tf.nn.relu,
                   name       = "",
                   data_format = "channels_first",
                   use_bias   = False,
                   regularize = 0.0,
                   bottleneck = 64,
                   reuse      = False):
    '''This is an all-in-one implementation of a residual block

    It can perform bottlenecking or not.
    
    
    Arguments:
        input_tensor {[type]} -- [description]
        is_training {bool} -- [description]
    
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

    with tf.variable_scope(name):

        x = input_tensor
        y = input_tensor

        # No matter what, this preserves the number of filters: 
        if data_format == "channels_last":
            n_output_filters = x.get_shape().as_list()[-1]
        else:
            n_output_filters = x.get_shape().as_list()[1]


        # If there is a bottle neck, there is an extra convolution:
        if bottleneck != -1:
            x = convolutional_block(x,
                        is_training,
                        n_filters  = bottleneck,
                        kernel     = (3,3),
                        strides    = (1,1),
                        batch_norm = batch_norm,
                        activation = tf.nn.relu,
                        data_format = data_format,
                        name       = "bottleneck1",
                        use_bias   = use_bias,
                        regularize = regularize,
                        reuse      = reuse)


        # Regardless of bottlenecking, we apply a main convolution
        # Setting n_filters = None preserves the number of filters
        x = convolutional_block(x,
                        is_training,
                        n_filters  = None,
                        kernel     = (3,3),
                        strides    = (1,1),
                        batch_norm = batch_norm,
                        activation = tf.nn.relu,
                        data_format = data_format,
                        name       = "main_op",
                        use_bias   = use_bias,
                        regularize = regularize,
                        reuse      = reuse)

        # The final convolution is either the anti-bottleneck or a standard convolution
        # Eitherway, the number of output filters is preserved from above.
        # NO ACTIVATION is applied here
        x = convolutional_block(x,
                        is_training,
                        n_filters  = n_output_filters,
                        kernel     = (3,3),
                        strides    = (1,1),
                        batch_norm = batch_norm,
                        activation = None,
                        data_format = data_format,
                        name       = "premerge_op",
                        use_bias   = use_bias,
                        regularize = regularize,
                        reuse      = reuse)

        # And finally, do the addition and the actication:
        x = x + y
        x = tf.nn.relu(x)

        return x







# Main class
class UResNet(object):
    '''Define a network model and run training

    resnet implementation
    '''
    def __init__(self):
        '''initialization

        Requires a list of parameters as python dictionary

        Arguments:
            params {dict} -- Network parameters

        Raises:
            ConfigurationException -- Missing a required parameter
        '''

        # Call the base class to initialize _core_network_params:

        # Extend the parameters to include the needed ones:

        return

    def _build_network(self, inputs):

        ''' FLAGS.VERBOSITY 0 = no printouts
            FLAGS.VERBOSITY 1 = sparse information
            FLAGS.VERBOSITY 2 = debug
        '''

        # Spin off the input image(s):

        x = inputs['image']

        if FLAGS.VERBOSITY > 1:
            sys.stdout.write("Initial input shape: {}\n".format(str(x.get_shape())))



        if FLAGS.SHARE_WEIGHTS:
            sharing = True
        else:
            sharing = False

        if FLAGS.VERBOSITY > 0:
            sys.stdout.write("Weight sharing across planes is: {}\n".format(str(sharing)))

        # We break up the intial filters into parallel U ResNets


        if FLAGS.VERBOSITY > 1:
            sys.stdout.write("Initial shape: {}\n".format(str(x.get_shape())))

        n_planes = FLAGS.NPLANES
        if FLAGS.DATA_FORMAT == "channels_first":
          channels_dim = 1
        else:
          channels_dim = -1


        if FLAGS.VERBOSITY > 1:
            sys.stdout.write("Attempting to split into {} planes\n".format(n_planes))
        x = tf.split(x, n_planes*[1], channels_dim)

        if FLAGS.VERBOSITY > 1:
            for p in range(len(x)):
                sys.stdout.write("Plane {0} initial shape: {1}\n".format(p, x[p].get_shape()))

        # Initial convolution to get to the correct number of filters:
        for p in range(len(x)):
            name = "Conv2DInitial"
            reuse = False
            if not sharing:
                name += "_plane{0}".format(p)
            if sharing and p != 0:
                reuse = True

            x[p] = convolutional_block(x[p],
                                       FLAGS.TRAINING,
                                       n_filters  = FLAGS.N_INITIAL_FILTERS,
                                       kernel     = (3,3),
                                       strides    = (1,1),
                                       batch_norm = FLAGS.BATCH_NORM,
                                       activation = tf.nn.relu,
                                       data_format= FLAGS.DATA_FORMAT,
                                       name       = name,
                                       use_bias   = FLAGS.USE_BIAS,
                                       regularize = FLAGS.REGULARIZE_WEIGHTS,
                                       reuse      = reuse)


            # ReLU:
            x[p] = tf.nn.relu(x[p])

        if FLAGS.VERBOSITY > 1:
            sys.stdout.write("After initial convolution: \n")

            for p in range(len(x)):
                sys.stdout.write("Plane {0}: {1}\n".format(p, x[p].get_shape()))



        if FLAGS.VERBOSITY > 0:
            sys.stdout.write("Begining downsampling\n")

        network_filters = [ [] for p in range(len(x)) ]

        # Begin the process of residual blocks and downsampling:
        for p in range(len(x)):
            for i in range(FLAGS.NETWORK_DEPTH):

                if FLAGS.CONNECT_PRE_RES_BLOCKS_DOWN:
                    network_filters[p].append(x[p])

                for j in range(FLAGS.RES_BLOCKS_PER_LAYER):
                    name = "resblock_down_{0}_{1}".format(i,j)
                    reuse = False
                    if not sharing:
                        name += "_plane{0}".format(p)
                    if sharing and p != 0:
                        reuse = True
                    x[p] = residual_block(x[p],
                                   FLAGS.TRAINING,
                                   strides    = (1,1),
                                   batch_norm = FLAGS.BATCH_NORM,
                                   activation = tf.nn.relu,
                                   data_format= FLAGS.DATA_FORMAT,
                                   name       = name,
                                   use_bias   = FLAGS.USE_BIAS,
                                   bottleneck = FLAGS.BOTTLENECK_SIZE,
                                   regularize = FLAGS.REGULARIZE_WEIGHTS,
                                   reuse      = reuse)

                if not FLAGS.CONNECT_PRE_RES_BLOCKS_DOWN:
                    network_filters[p].append(x[p])

                name = "downsample_{0}".format(i)
                reuse = False
                if not sharing:
                    name += "_plane{0}".format(p)
                if sharing and p != 0:
                    reuse = True
                # the downsample block doubles the number of filters:
                n_filters = 2*x[p].get_shape().as_list()[channels_dim]
                x[p] = convolutional_block(x[p],
                                   FLAGS.TRAINING,
                                   n_filters  = n_filters,
                                   kernel     = (2,2),
                                   strides    = (2,2),
                                   batch_norm = FLAGS.BATCH_NORM,
                                   activation = tf.nn.relu,
                                   data_format= FLAGS.DATA_FORMAT,
                                   name       = name,
                                   use_bias   = FLAGS.USE_BIAS,
                                   regularize = FLAGS.REGULARIZE_WEIGHTS,
                                   reuse      = reuse)


                if FLAGS.VERBOSITY > 1:
                    sys.stdout.write("Plane {p}, layer {i}: x[{p}].get_shape(): {s}\n".format(
                        p=p, i=i, s=x[p].get_shape()))

        # sys.stdout.write("Reached the deepest layer.")

        if FLAGS.VERBOSITY > 0:
            sys.stdout.write("Concatenating planes together\n")

        # Here, concatenate all the planes together before the residual block:
        x = tf.concat(x, axis=channels_dim)

        if FLAGS.VERBOSITY > 0:
            sys.stdout.write( "Shape after concat: {}\n".format(x.get_shape()))

        # At the bottom, do another residual block:
        for j in range(FLAGS.RES_BLOCKS_DEEPEST_LAYER):

            # Since all planes work together here, there is no weight sharing applied:
            reuse = False
            name = "deepest_res_block_{}".format(j)
            x = residual_block(x,
                               FLAGS.TRAINING,
                               strides    = (1,1),
                               batch_norm = FLAGS.BATCH_NORM,
                               activation = tf.nn.relu,
                               data_format= FLAGS.DATA_FORMAT,
                               name       = name,
                               use_bias   = FLAGS.USE_BIAS,
                               bottleneck = FLAGS.BOTTLENECK_SIZE,
                               regularize = FLAGS.REGULARIZE_WEIGHTS,
                               reuse      = reuse)


        # Need to split the network back into n_planes
        # The deepest block doesn't change the shape, so
        # it's easy to split:

        x = tf.split(x, n_planes, channels_dim)

        if FLAGS.VERBOSITY > 1:
            for p in range(len(x)):
                sys.stdout.write("Shape after split in plane {}: {}\n".format(p, x[p].get_shape()))



        # Come back up the network:
        for p in range(len(x)):
            for i in range(FLAGS.NETWORK_DEPTH-1, -1, -1):

                # sys.stdout.write( "Up start, Plane {p}, layer {i}: x[{p}].get_shape(): {s}".format(
                #     p=p, i=i, s=x[p].get_shape()))

                # How many filters to return from upsampling?
                n_filters = network_filters[p][-1].get_shape().as_list()[channels_dim]

                name = "upsample"
                reuse = False
                if not sharing:
                    name += "_plane{0}".format(p)
                if sharing and p != 0:
                    reuse = True

                name += "_{0}".format(i)
                if FLAGS.VERBOSITY > 1:
                    sys.stdout.write( "Name: {0} + reuse: {1}\n".format(name, reuse))

                # Upsample:
                x[p] = convolutional_upsample(x[p],
                        FLAGS.TRAINING,
                        n_filters  = n_filters,
                        kernel     = (2,2),
                        strides    = (2,2),
                        batch_norm = FLAGS.BATCH_NORM,
                        activation = tf.nn.relu,
                        data_format= FLAGS.DATA_FORMAT,
                        name       = name,
                        use_bias   = FLAGS.USE_BIAS,
                        regularize = FLAGS.REGULARIZE_WEIGHTS,
                        reuse      = reuse)

                if FLAGS.CONNECT_PRE_RES_BLOCKS_UP:
                    if FLAGS.CONNECTIONS == "sum":
                        x[p] = x[p] + network_filters[p][-1]
                    else:
                        n_filters = x[p].get_shape().as_list()[channels_dim]

                        x[p] = tf.concat([x[p], network_filters[p][-1]],
                                          axis=-1, name='up_concat_plane{0}_{1}'.format(p,i))
                        # Reshape with a bottleneck:
                        x[p] = convolutional_block(x[p],
                                                   FLAGS.TRAINING,
                                                   n_filters  = n_filters,
                                                   kernel     = (1,1),
                                                   strides    = (1,1),
                                                   batch_norm = FLAGS.BATCH_NORM,
                                                   activation = tf.nn.relu,
                                                   data_format= FLAGS.DATA_FORMAT,
                                                   name       = name,
                                                   use_bias   = FLAGS.USE_BIAS,
                                                   regularize = FLAGS.REGULARIZE_WEIGHTS,
                                                   reuse      = reuse)

                    # Remove the recently concated filters:
                    network_filters[p].pop()


                # Residual
                for j in range(FLAGS.RES_BLOCKS_PER_LAYER):
                    name = "resblock_up"
                    reuse = False
                    if not sharing:
                        name += "_plane{0}".format(p)
                    if sharing and p != 0:
                        reuse = True

                    name += "_{0}_{1}".format(i, j)

                    if FLAGS.VERBOSITY > 1:
                        sys.stdout.write( "Name: {0} + reuse: {1}\n".format(name, reuse))

                    x[p] = residual_block(x[p],
                                          FLAGS.TRAINING,
                                          strides    = (1,1),
                                          batch_norm = FLAGS.BATCH_NORM,
                                          activation = tf.nn.relu,
                                          data_format= FLAGS.DATA_FORMAT,
                                          name       = name,
                                          use_bias   = FLAGS.USE_BIAS,
                                          bottleneck = FLAGS.BOTTLENECK_SIZE,
                                          regularize = FLAGS.REGULARIZE_WEIGHTS,
                                          reuse      = reuse)


                if not FLAGS.CONNECT_PRE_RES_BLOCKS_UP:
                    if FLAGS.CONNECTIONS == "sum":
                        x[p] = x[p] + network_filters[p][-1]
                    else:
                        n_filters = x[p].get_shape().as_list()[-1]
                        x[p] = tf.concat([x[p], network_filters[p][-1]],
                                          axis=-1, name='up_concat_plane{0}_{1}'.format(p,i))
                        # Reshape with a bottleneck:
                        x[p] = convolutional_block(x[p],
                                                   FLAGS.TRAINING,
                                                   n_filters  = int(n_filters/2),
                                                   kernel     = (1,1),
                                                   strides    = (1,1),
                                                   batch_norm = FLAGS.BATCH_NORM,
                                                   data_format= FLAGS.DATA_FORMAT,
                                                   activation = tf.nn.relu,
                                                   name       = name,
                                                   use_bias   = FLAGS.USE_BIAS,
                                                   regularize = FLAGS.REGULARIZE_WEIGHTS,
                                                   reuse      = reuse)

                    # Remove the recently concated filters:
                    network_filters[p].pop()

                if FLAGS.VERBOSITY > 2:
                    sys.stdout.write( "Up end, Plane {p}, layer {i}: x[{p}].get_shape(): {s}\n".format(
                        p=p, i=i, s=x[p].get_shape()))

        # Here, the output is of the same size as the input but with the wrong number of filters
        # Apply any final residual blocks and then use a bottle neck to map to the right number
        # of filters
        for p in range(len(x)):
            for j in range(FLAGS.RES_BLOCKS_FINAL):
                name = "resblock_final"
                reuse = False
                if not sharing:
                    name += "_plane{0}".format(p)
                if sharing and p != 0:
                    reuse = True

                name += "_{0}".format(j)
                x[p] = residual_block(x[p],
                                      FLAGS.TRAINING,
                                      strides    = (1,1),
                                      batch_norm = FLAGS.BATCH_NORM,
                                      data_format= FLAGS.DATA_FORMAT,
                                      activation = tf.nn.relu,
                                      name       = name,
                                      use_bias   = FLAGS.USE_BIAS,
                                      bottleneck = FLAGS.BOTTLENECK_SIZE,
                                      regularize = FLAGS.REGULARIZE_WEIGHTS,
                                      reuse      = reuse)
            name = "final_bottleneck"
            reuse = False
            if not sharing:
                name += "_plane{0}".format(p)
            if sharing and p != 0:
                reuse = True
            # Map to the right number of labels (3) with a bottleneck:
            x[p] = convolutional_block(x[p],
                           FLAGS.TRAINING,
                           n_filters  = 3,
                           kernel     = (1,1),
                           strides    = (1,1),
                           batch_norm = FLAGS.BATCH_NORM,
                           data_format= FLAGS.DATA_FORMAT,
                           activation = None,
                           name       = name,
                           use_bias   = FLAGS.USE_BIAS,
                           regularize = FLAGS.REGULARIZE_WEIGHTS,
                           reuse      = reuse)




        logits = x

        return logits
