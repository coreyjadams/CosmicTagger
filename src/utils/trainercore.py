import os
import sys
import time
import tempfile
from collections import OrderedDict

import numpy

from larcv import queueloader


from . import flags
from . import data_transforms
from ..io import io_templates
from ..networks import uresnet
FLAGS = flags.FLAGS()

import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'


import tensorflow as tf

floating_point_format = tf.float32
integer_format = tf.int64


class trainercore(object):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self,):
        self._larcv_interface = queueloader.queue_interface()
        self._iteration       = 0
        self._global_step     = -1
        self._val_writer      = None

        self._cleanup         = []

    def __del__(self):
        for f in self._cleanup:
            try:
                os.unlink(f.name)
            except AttributeError:
                pass
            
    def _initialize_io(self, color=None):


        # This is a dummy placeholder, you must check this yourself:
        if 640 in FLAGS.SHAPE:
            max_voxels = 35000
        else:
            max_voxels = 70000

        # Use the templates to generate a configuration string, which we store into a temporary file
        if FLAGS.TRAINING:
            config = io_templates.train_io(
                input_file=FLAGS.FILE, 
                data_producer= FLAGS.IMAGE_PRODUCER,
                label_producer= FLAGS.LABEL_PRODUCER, 
                max_voxels=max_voxels)
        else:
            config = io_templates.ana_io(
                input_file=FLAGS.FILE, 
                data_producer= FLAGS.IMAGE_PRODUCER,
                label_producer= FLAGS.LABEL_PRODUCER, 
                max_voxels=max_voxels)


        # Generate a named temp file:
        main_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        main_file.write(config.generate_config_str())

        main_file.close()
        self._cleanup.append(main_file)

        # Prepare data managers:
        io_config = {
            'filler_name' : config._name,
            'filler_cfg'  : main_file.name,
            'verbosity'   : FLAGS.VERBOSITY,
            'make_copy'   : True
        }

        # By default, fetching data and label as the keywords from the file:
        data_keys = OrderedDict({
            'image': 'data', 
            'label': 'label'
            })


        self._larcv_interface.prepare_manager('primary', io_config, FLAGS.MINIBATCH_SIZE, data_keys, color)

        # All of the additional tools are in case there is a test set up:
        if FLAGS.AUX_FILE is not None:

            if FLAGS.TRAINING:
                config = io_templates.test_io(
                    input_file=FLAGS.AUX_FILE, 
                    data_producer= FLAGS.IMAGE_PRODUCER,
                    label_producer= FLAGS.LABEL_PRODUCER,
                    max_voxels=max_voxels)

                # Generate a named temp file:
                aux_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
                aux_file.write(config.generate_config_str())

                aux_file.close()
                self._cleanup.append(aux_file)
                io_config = {
                    'filler_name' : config._name,
                    'filler_cfg'  : aux_file.name,
                    'verbosity'   : FLAGS.VERBOSITY,
                    'make_copy'   : False
                }

                data_keys = OrderedDict({
                    'image': 'aux_data', 
                    'label': 'aux_label'
                    })
               


                self._larcv_interface.prepare_manager('aux', io_config, FLAGS.AUX_MINIBATCH_SIZE, data_keys, color)

            else:
                config = io_templates.ana_io(input_file=FLAGS.FILE, max_voxels=max_voxels)
                self._larcv_interface.prepare_writer(FLAGS.AUX_FILE)

    def init_network(self):

        # This function builds the compute graph.
        # Optionally, it can build a 'subset' graph if this mode is

        # Net construction:
        start = time.time()
        # sys.stdout.write("Begin constructing network\n")

        # Make sure all required dimensions are present:

        io_dims = self._larcv_interface.fetch_minibatch_dims('primary')

        if FLAGS.DATA_FORMAT == "channels_last":
            self._channels_dim = -1 
        else:
            self._channels_dim = 1

        self._dims = {}
        # Using the sparse IO techniques, we have to manually set the dimensions for the input.

        # Fortunately, everything we need is in the FLAGS object and io object:

        local_minibatch_size = io_dims['image'][0]


        if FLAGS.DATA_FORMAT == "channels_first":
            shape = [local_minibatch_size,] + [3,] + FLAGS.SHAPE
        else:
            shape = [local_minibatch_size,] + FLAGS.SHAPE + [3,]

        self._dims['image'] = numpy.asarray(shape)
        self._dims['label'] = numpy.asarray(shape)

        # We have to make placeholders for input objects:

        self._input = dict()

        self._input.update({
            'image' :  tf.placeholder(floating_point_format, self._dims['image'], name="input_image"),
            'label' :  tf.placeholder(integer_format,        self._dims['label'], name="input_label"),
            'io_time' : tf.placeholder(floating_point_format, (), name="io_fetch_time")
        })

        if FLAGS.BALANCE_LOSS:
            self._input['weight'] = tf.placeholder(floating_point_format, self._dims['label'], name="input_weight")

        # Build the network object, forward pass only:

        self._metrics = {}

        self._net = uresnet.UResNet(
            n_initial_filters        = FLAGS.N_INITIAL_FILTERS,
            data_format              = FLAGS.DATA_FORMAT,
            batch_norm               = FLAGS.BATCH_NORM,
            use_bias                 = FLAGS.USE_BIAS,
            residual                 = FLAGS.RESIDUAL,
            regularize               = FLAGS.REGULARIZE_WEIGHTS,
            depth                    = FLAGS.NETWORK_DEPTH,
            res_blocks_final         = FLAGS.RES_BLOCKS_FINAL,
            res_blocks_per_layer     = FLAGS.RES_BLOCKS_PER_LAYER,
            res_blocks_deepest_layer = FLAGS.RES_BLOCKS_DEEPEST_LAYER)

        self._logits = self._net(self._input['image'], training=FLAGS.TRAINING)


        if FLAGS.MODE == "train":


            # Here, if the data format is channels_first, we have to reorder the logits tensors
            # To put channels last.  Otherwise it does not work with the softmax tensors.

            # if FLAGS.DATA_FORMAT != "channels_last":
            #     # Split the channel dims apart:
            #     for i, logit in enumerate(self._logits):
            #         n_splits = logit.get_shape().as_list()[1]
                    
            #         # Split the tensor apart:
            #         split = [tf.squeeze(l, 1) for l in tf.split(logit, n_splits, 1)]
                    
            #         # Stack them back together with the right shape:
            #         self._logits[i] = tf.stack(split, -1)
            #         print
            # Apply a softmax and argmax:
            self._output = dict()

            # Take the logits (which are one per plane) and create a softmax and prediction (one per plane)

            self._output['softmax'] = [ tf.nn.softmax(x) for x in self._logits]
            self._output['prediction'] = [ tf.argmax(x, axis=self._channels_dim) for x in self._logits]


            self._accuracy = self._calculate_accuracy(logits=self._output, labels=self._input['label'])

            # Create the loss function
            if FLAGS.BALANCE_LOSS:
                self._loss = self._calculate_loss(
                    labels = self._input['label'], 
                    logits = self._logits, 
                    weight = self._input['weight'])
            else:
                self._loss = self._calculate_loss(
                        labels = self._input['label'], 
                        logits = self._logits)

        self._log_keys = ["cross_entropy/Total_Loss", "accuracy/All_Plane_Non_Background_Accuracy"]

        end = time.time()
        return end - start

    def print_network_info(self):
        n_trainable_parameters = 0
        for var in tf.trainable_variables():
            n_trainable_parameters += numpy.prod(var.get_shape())
        sys.stdout.write("Total number of trainable parameters in this network: {}\n".format(n_trainable_parameters))


    def set_compute_parameters(self):

        self._config = tf.ConfigProto()

        if FLAGS.COMPUTE_MODE == "CPU":
            self._config.inter_op_parallelism_threads = FLAGS.INTER_OP_PARALLELISM_THREADS
            self._config.intra_op_parallelism_threads = FLAGS.INTRA_OP_PARALLELISM_THREADS
        if FLAGS.COMPUTE_MODE == "GPU":
            self._config.gpu_options.allow_growth = True


    def initialize(self, io_only=False):

        FLAGS.dump_config()


        self._initialize_io()



        if io_only:
            return


        start = time.time()
        graph = tf.get_default_graph()
        net_time = self.init_network()

        sys.stdout.write("Done constructing network. ({0:.2}s)\n".format(time.time()-start))


        self.print_network_info()


        self.init_optimizer()

        self.init_saver()

        # Take all of the metrics and turn them into summaries:
        for key in self._metrics:
            tf.summary.scalar(key, self._metrics[key])

        self._summary_basic = tf.summary.merge_all()
        self._summary_images = self._create_summary_images(self._input['label'], self._output['prediction'])


        self.set_compute_parameters()

        # Add the graph to the log file:
        self._main_writer.add_graph(graph)

        self._sess = tf.Session(config = self._config)

        # Try to restore a model?
        restored = self.restore_model()

        if not restored:
            self._sess.run(tf.global_variables_initializer())

        # # Create a session:
        # self._sess = tf.train.MonitoredTrainingSession(config=self._config, hooks = hooks,
        #     checkpoint_dir        = checkpoint_dir,
        #     log_step_count_steps  = FLAGS.LOGGING_ITERATION,
        #     save_checkpoint_steps = FLAGS.CHECKPOINT_ITERATION)


    def restore_model(self):
        ''' This function attempts to restore the model from file
        '''

        if FLAGS.CHECKPOINT_DIRECTORY == None:
            file_path= FLAGS.LOG_DIRECTORY  + "/checkpoints/"
        else:
            file_path= FLAGS.CHECKPOINT_DIRECTORY  + "/checkpoints/"


        path = tf.train.latest_checkpoint(file_path)

        if path is None:
            print("No checkpoint found, starting from scratch")
            return False
        # Parse the checkpoint file and use that to get the latest file path
        print("Restoring checkpoint from ", path)
        self._saver.restore(self._sess, path)

        return True

        # with open(checkpoint_file_path, 'r') as _ckp:
        #     for line in _ckp.readlines():
        #         if line.startswith("latest: "):
        #             chkp_file = line.replace("latest: ", "").rstrip('\n')
        #             chkp_file = os.path.dirname(checkpoint_file_path) + "/" + chkp_file
        #             print("Restoring weights from ", chkp_file)
        #             break

        # state = torch.load(chkp_file)
        # return state


    def checkpoint(self, global_step):

        if global_step % FLAGS.CHECKPOINT_ITERATION == 0 and global_step != 0:
            # Save a checkpoint, but don't do it on the first pass
            self.save_model(global_step)


    def save_model(self, global_step):
        '''Save the model to file
        
        '''

        # name, checkpoint_file_path = self.get_model_filepath(global_step)
        # Find the base path of the log directory
        if FLAGS.CHECKPOINT_DIRECTORY == None:
            file_path= FLAGS.LOG_DIRECTORY  + "/checkpoints/"
        else:
            file_path= FLAGS.CHECKPOINT_DIRECTORY  + "/checkpoints/"


        # # Make sure the path actually exists:
        # if not os.path.isdir(os.path.dirname(file_path)):
        #     os.makedirs(os.path.dirname(file_path))

        saved_path = self._saver.save(self._sess, file_path + "model_{}.ckpt".format(global_step))


    def get_model_filepath(self, global_step):
        '''Helper function to build the filepath of a model for saving and restoring:
        
        '''

        # Find the base path of the log directory
        if FLAGS.CHECKPOINT_DIRECTORY == None:
            file_path= FLAGS.LOG_DIRECTORY  + "/checkpoints/"
        else:
            file_path= FLAGS.CHECKPOINT_DIRECTORY  + "/checkpoints/"


        name = file_path + 'model-{}.ckpt'.format(global_step)
        checkpoint_file_path = file_path + "checkpoint"

        return name, checkpoint_file_path


    def init_saver(self):

        if FLAGS.CHECKPOINT_DIRECTORY == None:
            file_path= FLAGS.LOG_DIRECTORY  + "/checkpoints/"
        else:
            file_path= FLAGS.CHECKPOINT_DIRECTORY  + "/checkpoints/"

        try:
            os.makedirs(file_path)
        except:
            tf.logging.error("Could not make file path")

        # Create a saver for snapshots of the network:
        self._saver = tf.train.Saver()
        self._saver_dir = file_path

        # Create a file writer for training metrics:
        self._main_writer = tf.summary.FileWriter(logdir=FLAGS.LOG_DIRECTORY+"/train/")

        # Additionally, in training mode if there is aux data use it for validation:
        if FLAGS.AUX_FILE is not None:
            self._val_writer = tf.summary.FileWriter(logdir=FLAGS.LOG_DIRECTORY+"/test/")



    def init_optimizer(self):

        if 'RMS' in FLAGS.OPTIMIZER.upper():
            # Use RMS prop:
            tf.logging.info("Selected optimizer is RMS Prop")
            self._opt = tf.train.RMSPropOptimizer(FLAGS.LEARNING_RATE)
        elif 'LARS' in FLAGS.OPTIMIZER.upper():
            tf.logging.info("Selected optimizer is LARS")
            self._opt = tf.contrib.opt.LARSOptimizer(FLAGS.LEARNING_RATE)
        else:
            # default is Adam:
            tf.logging.info("Using default Adam optimizer")
            self._opt = tf.train.AdamOptimizer(FLAGS.LEARNING_RATE)

        self._global_step = tf.train.get_or_create_global_step()


        self._train_op = self._opt.minimize(self._loss, self._global_step)


    def _calculate_loss(self, labels, logits, weight=None):
        ''' Calculate the loss.

        returns a single scalar for the optimizer to use.
        '''        

        with tf.name_scope('cross_entropy'):
            # Calculate the loss, per plane, unreduced:
            split_labels = [tf.squeeze(l, axis=self._channels_dim) for l in tf.split(labels,len(logits) ,self._channels_dim)]
            if weight is not None:
                split_weights = [tf.squeeze(l, axis=self._channels_dim) for l in tf.split(weight,len(logits) ,self._channels_dim)]
            
            
            # If the channels dim is not -1, we have to reshape the labels:
            if self._channels_dim != -1:
                logits = [ tf.transpose(l, perm=[0,2,3,1]) for l in logits]

            loss = [None]*len(logits)
            for p in range(len(logits)):
                loss[p] = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels = split_labels[p], 
                    logits = logits[p]
                )

                # multiple (elementwise) the weights for the loss function:
                if FLAGS.BALANCE_LOSS:
                    loss[p] = tf.multiply(loss[p], split_weights[p])
                    # Because we have a weighting function, this is a summed reduction:
                    loss[p] = tf.reduce_sum(loss[p])
                else:
                    loss[p] = tf.reduce_mean(loss[p])                

                self._metrics["cross_entropy/Loss_plane_{}".format(p)] = loss[p]
                # tf.summary.scalar("Loss_plane_{}".format(p),loss[p])

            # We do use the *mean* across planes:
            total_loss = tf.reduce_mean(loss)

            # # If desired, add weight regularization loss:
            # if FLAGS.REGULARIZE_WEIGHTS != 0.0: 
            #     reg_loss = tf.reduce_mean(tf.losses.get_regularization_losses())
            #     tf.summary.scalar("Regularization_loss",reg_loss)
            #     total_loss += reg_loss


            # Total summary:
            self._metrics["cross_entropy/Total_Loss"] = total_loss
            # tf.summary.scalar("Total_Loss",total_loss)

            return total_loss


    def _calculate_accuracy(self, logits, labels):
        ''' Calculate the accuracy.

            Images received here are not sparse but dense.
            This is to ensure equivalent metrics are computed for sparse and dense networks.

        '''

        # Compare how often the input label and the output prediction agree:

        n_planes = 3

        with tf.name_scope('accuracy'):
            total_accuracy   = [None]*n_planes
            non_bkg_accuracy = [None]*n_planes
            neut_iou         = [None]*n_planes
            cosmic_iou       = [None]*n_planes
            
            split_labels = [tf.squeeze(l, axis=self._channels_dim) for l in tf.split(labels, n_planes, self._channels_dim)]


            for p in range(n_planes):

                total_accuracy[p] = tf.reduce_mean(
                        tf.cast(tf.equal(logits['prediction'][p], split_labels[p]), floating_point_format)
                    )
                # Find the non zero split_labels:
                non_zero_indices = tf.not_equal(split_labels[p], tf.constant(0, split_labels[p].dtype))

                # Find the neutrino indices:
                # Sometimes, there are no neutrino indexes in the image.  This leads to a NaN
                # in the calculation of the neutrino accuracy.  
                # This is an open issue to resolve.
                neutrino_indices = tf.equal(split_labels[p], tf.constant(1, split_labels[p].dtype))

                # Find the cosmic indices:
                cosmic_indices = tf.equal(split_labels[p], tf.constant(2, split_labels[p].dtype))

                non_zero_logits = tf.boolean_mask(logits['prediction'][p], non_zero_indices)
                non_zero_labels = tf.boolean_mask(split_labels[p], non_zero_indices)

                neutrino_logits = tf.boolean_mask(logits['prediction'][p], neutrino_indices)
                neutrino_labels = tf.boolean_mask(split_labels[p], neutrino_indices)

                predicted_neutrino_indices = tf.equal(logits['prediction'][p], 
                    tf.constant(1, split_labels[p].dtype))
                predicted_cosmic_indices = tf.equal(logits['prediction'][p], 
                    tf.constant(2, split_labels[p].dtype))


                neutrino_intersection = tf.math.logical_and(predicted_neutrino_indices, neutrino_indices)
                neutrino_union = tf.math.logical_or(predicted_neutrino_indices, neutrino_indices)

                neut_iou[p] = tf.reduce_sum(tf.cast(neutrino_intersection, floating_point_format)) / \
                  (tf.reduce_sum(tf.cast(neutrino_union, floating_point_format)) + 1.0)

                cosmic_intersection = tf.math.logical_and(predicted_cosmic_indices, cosmic_indices)
                cosmic_union = tf.math.logical_or(predicted_cosmic_indices, cosmic_indices)

                cosmic_iou[p] = tf.reduce_sum(tf.cast(cosmic_intersection, floating_point_format)) / \
                  tf.reduce_sum(tf.cast(cosmic_union, floating_point_format))

                non_bkg_accuracy[p] = tf.reduce_mean(tf.cast(tf.equal(non_zero_logits, non_zero_labels), 
                    floating_point_format))

                # Add the accuracies to the summary:
                self._metrics["split_accuracy/plane{0}/Total_Accuracy".format(p)] = total_accuracy[p]
                self._metrics["split_accuracy/plane{0}/Non_Background_Accuracy".format(p)] = non_bkg_accuracy[p]
                self._metrics["split_accuracy/plane{0}/Neutrino_IoU".format(p)] = neut_iou[p]
                self._metrics["split_accuracy/plane{0}/Cosmic_IoU".format(p)] = cosmic_iou[p]

            #Compute the total accuracy and non background accuracy for all planes:
            all_accuracy            = tf.reduce_mean(total_accuracy)
            all_non_bkg_accuracy    = tf.reduce_mean(non_bkg_accuracy)
            all_neut_iou            = tf.reduce_mean(neut_iou)
            all_cosmic_iou          = tf.reduce_mean(cosmic_iou)

            # Add the accuracies to the summary:
            self._metrics["accuracy/All_Plane_Total_Accuracy"] = all_accuracy
            self._metrics["accuracy/All_Plane_Non_Background_Accuracy"] = all_non_bkg_accuracy
            self._metrics["accuracy/All_Plane_Neutrino_IoU"] = all_neut_iou
            self._metrics["accuracy/All_Plane_Cosmic_IoU"] = all_cosmic_iou


        return all_non_bkg_accuracy



    def log(self, metrics, kind, step):

        log_string = ""

        log_string += "{} Global Step {}: ".format(kind, step)


        for key in metrics:
            if key in self._log_keys and key != "global_step":
                log_string += "{}: {:.3}, ".format(key, metrics[key])

        if kind == "Train":
            log_string += "Img/s: {:.2} ".format(metrics["images_per_second"])
            log_string += "IO: {:.2} ".format(metrics["io_fetch_time"])
        else:
            log_string.rstrip(", ")

        print(log_string)

        return


    def _create_summary_images(self, labels, prediction):
        ''' Create images of the labels and prediction to show training progress
        '''

        with tf.variable_scope('summary_images/'):


            images = []

            # Labels is an unsplit tensor, prediction is a split tensor
            split_labels = [ tf.cast(l, floating_point_format) for l in tf.split(labels,len(prediction) , self._channels_dim)]
            if FLAGS.DATA_FORMAT == "channels_first":
                split_labels = [ tf.transpose(l, [0, 2, 3, 1]) for l in split_labels]
                print("split_labels[0].shape: ", split_labels[0].shape)
            prediction = [ tf.expand_dims(tf.cast(p, floating_point_format), self._channels_dim) for p in prediction ]
            print("prediction[0].shape: ", prediction[0].shape)

            for p in range(len(split_labels)):
                
                images.append(
                    tf.summary.image('label_plane_{}'.format(p),
                                 split_labels[p],
                                 max_outputs=1)
                    )
                images.append(
                    tf.summary.image('pred_plane_{}'.format(p),
                                 prediction[p],
                                 max_outputs=1)
                    )

        return tf.summary.merge(images)

    def fetch_next_batch(self, mode='primary', metadata=False):


        metadata=True
        self._larcv_interface.prepare_next(mode)

        # This brings up the current data
        minibatch_data = self._larcv_interface.fetch_minibatch_data(mode, pop=True,fetch_meta_data=metadata)
        minibatch_dims = self._larcv_interface.fetch_minibatch_dims(mode)


        for key in minibatch_data:
            if key == 'entries' or key == 'event_ids':
                continue
            minibatch_data[key] = numpy.reshape(minibatch_data[key], minibatch_dims[key])

        if FLAGS.BALANCE_LOSS:
            minibatch_data['weight'] = self.compute_weights(minibatch_data['label'])

            print(minibatch_data['weight'].shape)

        minibatch_data['image']  = data_transforms.larcvsparse_to_dense_2d(minibatch_data['image'], dense_shape=FLAGS.SHAPE)
        minibatch_data['label']  = data_transforms.larcvsparse_to_dense_2d(minibatch_data['label'], dense_shape=FLAGS.SHAPE)
        # This preparse the next batch of data:



        return minibatch_data


    def compute_weights(self, labels, boost_labels = None):
        '''
        This is NOT a tensorflow implementation, but a numpy implementation.
        Running on CPUs this might not make a difference.  Running on GPUs
        it might be good to move this to a GPU, but I suspect it's not needed.
        '''
        # Take the labels, and compute the per-label weight

        # Compute weights works on the sparse images, not the dense images.
        # The null-weight is computed on the image shape for dense networks, 
        # or based on occupancy on the sparse network.

        # It's done per-batch rather than per image, so:

        x_coords = labels[:,:,:,1]
        y_coords = labels[:,:,:,0]
        val_coords = labels[:,:,:,2]


        # Find the non_zero indexes of the input:
        batch_index, plane_index, voxel_index = numpy.where(val_coords != -999)

        values  = val_coords[batch_index, plane_index, voxel_index]
        x_index = numpy.int32(x_coords[batch_index, plane_index, voxel_index])
        y_index = numpy.int32(y_coords[batch_index, plane_index, voxel_index])

        label_values, counts = numpy.unique(values, return_counts=True)

        if len(counts) < 3:
            counts = numpy.insert(counts, 1, 0)

        batch_size = labels.shape[0]



        if not FLAGS.SPARSE:
            # Multiply by 3 planes:
            n_pixels = batch_size * 3* numpy.prod(FLAGS.SHAPE)
            # Correct the empty pixel values in the count:
            counts[0] = n_pixels - counts[1] - counts[2]
        else:
            n_pixels = len(values)

        weight = 1.0/ (len(label_values) * counts)


        # Now we have the weight values, return it in the proper shape:
        # Prepare output weights:
        weights = numpy.full(values.shape, weight[0])
        weights[voxel_index==1] = weight[1]
        weights[voxel_index==2] = weight[2]

        dense_weights = numpy.full([labels.shape[0], 3, FLAGS.SHAPE[0], FLAGS.SHAPE[1]], weight[0])
        dense_weights[batch_index,plane_index,y_index,x_index] = weights

        # i = 0
        # for batch in labels:
        #     # First, figure out what the labels are and how many of each:
        #     values, counts = numpy.unique(batch, return_counts=True)

        #     n_pixels = numpy.sum(counts)
        #     for value, count in zip(values, counts):
        #         weight = 1.0*(n_pixels - count) / n_pixels
        #         mask = labels[i] == value
        #         weights[i, mask] += weight
        #     weights[i] *= 1. / numpy.sum(weights[i])
        #     i += 1



        # # Normalize the weights to sum to 1 for each event:
        return dense_weights

    def on_step_end(self):
        pass

    def on_epoch_end(self):
        pass

    def write_summaries(self, writer, summary, global_step):
        # This function is isolated here to allow the distributed version
        # to intercept these calls and only write summaries from one rank

        writer.add_summary(summary, global_step)

    def metrics(self, metrics):
        # This function looks useless, but it is not.
        # It allows a handle to the distributed network to allreduce metrics.
        return metrics

    def val_step(self, gs):

        if gs == 0: return

        if self._val_writer is None:
            return

        if gs % FLAGS.AUX_ITERATION == 0:


            # Fetch the next batch of data with larcv
            minibatch_data = self.fetch_next_batch('aux')

            # For tensorflow, we have to build up an ops list to submit to the
            # session to run.

            # These are ops that always run:
            ops = {}
            ops['global_step'] = self._global_step
            ops['summary'] = self._summary_basic

            ops['metrics'] = self._metrics

            if self._iteration != 0 and self._iteration % 50*FLAGS.SUMMARY_ITERATION == 0:
                ops['summary_images'] = self._summary_images


            ops = self._sess.run(ops, feed_dict = self.feed_dict(inputs = minibatch_data))

            metrics = self.metrics(ops["metrics"])

            verbose = False




            if verbose: print("Calculated metrics")

            # Report metrics on the terminal:
            self.log(ops["metrics"], kind="Test", step=ops["global_step"]) 


            if verbose: print("Completed Log")

            self.write_summaries(self._val_writer, ops["summary"], ops["global_step"])
            if self._iteration != 0 and self._iteration % 50*FLAGS.SUMMARY_ITERATION == 0:
                self.write_summaries(self._val_writer, ops["summary_images"], ops["global_step"])


            if verbose: print("Summarized")


            # Lastly, call next on the IO:
            if not FLAGS.DISTRIBUTED:
                self._larcv_interface.prepare_next('aux')

            return ops["global_step"]
        return


    def train_step(self):


        global_start_time = datetime.datetime.now()

        # Fetch the next batch of data with larcv
        io_start_time = datetime.datetime.now()
        minibatch_data = self.fetch_next_batch()
        io_end_time = datetime.datetime.now()

        # For tensorflow, we have to build up an ops list to submit to the
        # session to run.

        # These are ops that always run:
        ops = {}
        ops['train_step']  = self._train_op
        ops['global_step'] = self._global_step
        ops['summary'] = self._summary_basic

        ops['metrics'] = self._metrics

        if self._iteration != 0 and self._iteration % 50*FLAGS.SUMMARY_ITERATION == 0:
            ops['summary_images'] = self._summary_images


        ops = self._sess.run(ops, feed_dict = self.feed_dict(inputs = minibatch_data))




        metrics = self.metrics(ops["metrics"])

        verbose = False

        # Add the global step / second to the tensorboard log:
        try:
            metrics['global_step_per_sec'] = 1./self._seconds_per_global_step
            metrics['images_per_second'] = FLAGS.MINIBATCH_SIZE / self._seconds_per_global_step
        except AttributeError:
            metrics['global_step_per_sec'] = 0.0
            metrics['images_per_second'] = 0.0



        metrics['io_fetch_time'] = (io_end_time - io_start_time).total_seconds()

        if verbose: print("Calculated metrics")

        # Report metrics on the terminal:
        self.log(ops["metrics"], kind="Train", step=ops["global_step"]) 


        if verbose: print("Completed Log")

        self.write_summaries(self._main_writer, ops["summary"], ops["global_step"])
        if self._iteration != 0 and self._iteration % 50*FLAGS.SUMMARY_ITERATION == 0:
            self.write_summaries(self._main_writer, ops["summary_images"], ops["global_step"])


        # Create some extra summary information:
        extra_summary = tf.Summary(
            value=[
                tf.Summary.Value(tag="io_fetch_time", simple_value=metrics['io_fetch_time']),
                tf.Summary.Value(tag="global_step_per_sec", simple_value=metrics['global_step_per_sec']),
                tf.Summary.Value(tag="images_per_second", simple_value=metrics['images_per_second']),
            ])

        self.write_summaries(self._main_writer, extra_summary, ops["global_step"])

        if verbose: print("Summarized")

        global_end_time = datetime.datetime.now()

        # Compute global step per second:
        self._seconds_per_global_step = (global_end_time - global_start_time).total_seconds()

        # Lastly, call next on the IO:
        if not FLAGS.DISTRIBUTED:
            self._larcv_interface.prepare_next('primary')

        return ops["global_step"]




 
    def stop(self):
        # Mostly, this is just turning off the io:
        # self._larcv_interface.stop()
        pass


    def ana_step(self):

        raise NotImplementedError("You must implement this function")

    def feed_dict(self, inputs):
        '''Build the feed dict

        Take input images, labels and match
        to the correct feed dict tensorrs

        This is probably overridden in the subclass, but here you see the idea

        Arguments:
            images {dict} -- Dictionary containing the input tensors

        Returns:
            [dict] -- Feed dictionary for a tf session run call

        '''
        fd = dict()

        for key in inputs:
            if key == "entries" or key == "event_ids": continue 
            
            if inputs[key] is not None:
                fd.update({self._input[key] : inputs[key]})

        return fd


    def batch_process(self, verbose=True):

        # Run iterations
        for self._iteration in range(FLAGS.ITERATIONS):
            if FLAGS.TRAINING and self._iteration >= FLAGS.ITERATIONS:
                print('Finished training (iteration %d)' % self._iteration)
                break

            gs = self.train_step()
            self.val_step(gs)
            self.checkpoint(gs)
