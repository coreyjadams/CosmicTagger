import os
import sys
import time
import tempfile
from collections import OrderedDict

import numpy

from larcv import larcv_interface


from . import flags
from . import data_transforms
from ..io import io_templates
FLAGS = flags.FLAGS()

import datetime
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
logging.getLogger("tensorflow").setLevel(logging.WARNING)

import tensorflow as tf

class trainercore(object):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self,):
        self._larcv_interface = larcv_interface.larcv_interface()
        self._iteration       = 0
        self._global_step     = -1
        self._val_writer      = None

        self._cleanup         = []

    def __del__(self):
        for f in self._cleanup:
            os.unlink(f.name)
            
    def _initialize_io(self):


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



        self._larcv_interface.prepare_manager('primary', io_config, FLAGS.MINIBATCH_SIZE, data_keys)

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
                    'make_copy'   : True
                }

                data_keys = OrderedDict({
                    'image': 'aux_data', 
                    'label': 'aux_label'
                    })
               


                self._larcv_interface.prepare_manager('aux', io_config, FLAGS.AUX_MINIBATCH_SIZE, data_keys)

            else:
                config = io_templates.ana_io(input_file=FLAGS.FILE, max_voxels=max_voxels)
                self._larcv_interface.prepare_writer(FLAGS.AUX_FILE)

    def init_network(self):

        # This function builds the compute graph.
        # Optionally, it can build a 'subset' graph if this mode is

        # Net construction:
        start = time.time()
        sys.stdout.write("Begin constructing network\n")

        # Make sure all required dimensions are present:

        io_dims = self._larcv_interface.fetch_minibatch_dims('primary')

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
            'image' :  tf.placeholder(tf.float32, self._dims['image'], name="input_image"),
            'label' :  tf.placeholder(tf.int64,   self._dims['label'], name="input_label"),
            'io_time' : tf.placeholder(tf.float32, (), name="io_fetch_time")
        })

        if FLAGS.BALANCE_LOSS:
            self._input['weight'] = tf.placeholder(tf.float32, self._dims['label'], name="input_weight"),


        # Build the network object, forward pass only:

        self._metrics = {}

        self._logits = FLAGS._net._build_network(self._input)


        if FLAGS.MODE == "train":


            # Here, if the data format is channels_first, we have to reorder the logits tensors
            # To put channels last.  Otherwise it does not work with the softmax tensors.

            if FLAGS.DATA_FORMAT != "channels_last":
                # Split the channel dims apart:
                for i, logit in enumerate(self._logits):
                    n_splits = logit.get_shape().as_list()[1]
                    
                    # Split the tensor apart:
                    split = [tf.squeeze(l, 1) for l in tf.split(logit, n_splits, 1)]
                    
                    # Stack them back together with the right shape:
                    self._logits[i] = tf.stack(split, -1)

            # Apply a softmax and argmax:
            self._output = dict()

            # Take the logits (which are one per plane) and create a softmax and prediction (one per plane)

            self._output['softmax'] = [ tf.nn.softmax(x) for x in self._logits]
            self._output['prediction'] = [ tf.argmax(x, axis=-1) for x in self._logits]



            self._accuracy = self._calculate_accuracy(logits=self._output, labels=self._input['label'])

            # Create the loss function
            self._loss = self._calculate_loss(
                labels = self._input['label'], 
                logits = self._logits, 
                weight = self._input['weight'])
            
            self._summary_images = self._create_summary_images(self._input['label'], self._output['prediction'])

        end = time.time()
        sys.stdout.write("Done constructing network. ({0:.2}s)\n".format(end-start))

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

        graph = tf.get_default_graph()
        self.init_network()

        self.print_network_info()

        self.init_optimizer()

        # Additionally, in training mode if there is aux data use it for validation:
        if FLAGS.AUX_FILE is not None:
            self._val_writer = tf.summary.FileWriter(logdir=FLAGS.LOG_DIRECTORY+"/test/")

        # Take all of the metrics and turn them into summaries:
        for key in self._metrics:
            tf.summary.scalar(key, self._metrics[key])

        self._summary_basic = tf.summary.merge_all()


        self.set_compute_parameters()

        hooks = self.get_standard_hooks()

        if FLAGS.CHECKPOINT_DIRECTORY is not None:
            checkpoint_dir = FLAGS.CHECKPOINT_DIRECTORY
        else:
            checkpoint_dir = FLAGS.LOG_DIRECTORY


        # Create a session:
        self._sess = tf.train.MonitoredTrainingSession(config=self._config, hooks = hooks,
            checkpoint_dir        = checkpoint_dir,
            log_step_count_steps  = FLAGS.LOGGING_ITERATION,
            save_checkpoint_steps = FLAGS.CHECKPOINT_ITERATION)



    def get_standard_hooks(self):

        hooks = []

        loss_is_nan_hook = tf.train.NanTensorHook(
            self._loss,
            fail_on_nan_loss=True,
        )
        hooks.append(loss_is_nan_hook)

        # Create a hook to manage the summary saving:
        summary_saver_hook = tf.train.SummarySaverHook(
            save_steps = FLAGS.SUMMARY_ITERATION,
            output_dir = FLAGS.LOG_DIRECTORY,
            summary_op = self._summary_basic
            )
        hooks.append(summary_saver_hook)

        summary_saver_hook_image = tf.train.SummarySaverHook(
            save_steps = 10*FLAGS.SUMMARY_ITERATION,
            output_dir = FLAGS.LOG_DIRECTORY,
            summary_op = self._summary_images
            )
        hooks.append(summary_saver_hook_image)

        # if FLAGS.PROFILE_ITERATION != -1:
        #     # Create a profiling hook for tracing:
        #     profile_hook = tf.train.ProfilerHook(
        #         save_steps    = FLAGS.PROFILE_ITERATION,
        #         output_dir    = FLAGS.LOG_DIRECTORY,
        #         show_dataflow = True,
        #         show_memory   = True
        #     )
        #     hooks.append(profile_hook)

        logging_hook = tf.train.LoggingTensorHook(
            tensors       = { 'global_step' : self._global_step,
                              'accuracy'    : self._accuracy,
                              'loss'        : self._loss},
            every_n_iter  = FLAGS.LOGGING_ITERATION,
            )
        hooks.append(logging_hook)


        return hooks

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


    def _calculate_loss(self, labels, logits, weight):
        ''' Calculate the loss.

        returns a single scalar for the optimizer to use.
        '''        

        if FLAGS.DATA_FORMAT == "channels_last":
            channels_dim = -1 
        else:
            channels_dim = 1

        with tf.name_scope('cross_entropy'):

            # Calculate the loss, per plane, unreduced:
            split_labels = [tf.squeeze(l, axis=channels_dim) for l in tf.split(labels,len(logits) ,channels_dim)]
            split_weights = [tf.squeeze(l, axis=channels_dim) for l in tf.split(weight,len(logits) ,channels_dim)]
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
        if FLAGS.DATA_FORMAT == "channels_last":
            channels_dim = -1 
        else:
            channels_dim = 1

        with tf.name_scope('accuracy'):
            total_accuracy   = [None]*len(logits['prediction'])
            non_bkg_accuracy = [None]*len(logits['prediction'])
            neut_accuracy    = [None]*len(logits['prediction'])
            
            split_labels = [tf.squeeze(l, axis=channels_dim) for l in tf.split(labels,len(logits['prediction']) , channels_dim)]

            for p in range(len(logits['prediction'])):

                total_accuracy[p] = tf.reduce_mean(
                        tf.cast(tf.equal(logits['prediction'][p], split_labels[p]), tf.float32)
                    )
                # Find the non zero split_labels:
                non_zero_indices = tf.not_equal(split_labels[p], tf.constant(0, split_labels[p].dtype))

                # Find the neutrino indices:
                neutrino_indices = tf.equal(split_labels[p], tf.constant(1, split_labels[p].dtype))

                non_zero_logits = tf.boolean_mask(logits['prediction'][p], non_zero_indices)
                non_zero_labels = tf.boolean_mask(split_labels[p], non_zero_indices)

                neutrino_logits = tf.boolean_mask(logits['prediction'][p], neutrino_indices)
                neutrino_labels = tf.boolean_mask(split_labels[p], neutrino_indices)

                non_bkg_accuracy[p] = tf.reduce_mean(tf.cast(tf.equal(non_zero_logits, non_zero_labels), tf.float32))
                neut_accuracy[p]    = tf.reduce_mean(tf.cast(tf.equal(neutrino_logits, neutrino_labels), tf.float32))

                # Add the accuracies to the summary:
                self._metrics["accuracy/Total_Accuracy_plane{0}".format(p)] = total_accuracy[p]
                # tf.summary.scalar("Total_Accuracy_plane{0}".format(p),
                #     total_accuracy[p])
                self._metrics["accuracy/Non_Background_Accuracy_plane{0}".format(p)] = non_bkg_accuracy[p]
                # tf.summary.scalar("Non_Background_Accuracy_plane{0}".format(p),
                #     non_bkg_accuracy[p])
                self._metrics["accuracy/Neutrino_Accuracy_plane{0}".format(p)] = neut_accuracy[p]
                # tf.summary.scalar("Neutrino_Accuracy_plane{0}".format(p),
                #     neut_accuracy[p])

            #Compute the total accuracy and non background accuracy for all planes:
            all_accuracy            = tf.reduce_mean(total_accuracy)
            all_non_bkg_accuracy    = tf.reduce_mean(non_bkg_accuracy)
            all_neut_accuracy       = tf.reduce_mean(neut_accuracy)

            # Add the accuracies to the summary:
            self._metrics["accuracy/All_Plane_Total_Accuracy"] = all_accuracy
            # tf.summary.scalar("All_Plane_Total_Accuracy", all_accuracy)
            self._metrics["accuracy/All_Plane_Non_Background_Accuracy"] = all_non_bkg_accuracy
            # tf.summary.scalar("All_Plane_Non_Background_Accuracy", all_non_bkg_accuracy)
            self._metrics["accuracy/All_Plane_Neutrino_Accuracy"] = all_neut_accuracy
            # tf.summary.scalar("All_Plane_Neutrino_Accuracy", all_neut_accuracy)


        return all_non_bkg_accuracy

    def _compute_metrics(self, logits, labels, loss):

        raise NotImplementedError("You must implement this function")
        


    def log(self, metrics, saver=''):

        raise NotImplementedError("You must implement this function")


    def summary(self, metrics,saver=""):
        

        raise NotImplementedError("You must implement this function")


    def _create_summary_images(self, labels, prediction):
        ''' Create images of the labels and prediction to show training progress
        '''
        if FLAGS.DATA_FORMAT == "channels_last":
            channels_dim = -1 
        else:
            channels_dim = 1

        with tf.variable_scope('summary_images/'):


            images = []

            # Labels is an unsplit tensor, prediction is a split tensor
            split_labels = [ tf.cast(l, tf.float32) for l in tf.split(labels,len(prediction) , channels_dim)]
            for p in range(len(split_labels)):
                images.append(
                    tf.summary.image('label_plane_{}'.format(p),
                                 split_labels[p],
                                 max_outputs=1)
                    )
                images.append(
                    tf.summary.image('pred_plane_{}'.format(p),
                                 tf.expand_dims(tf.cast(prediction[p], tf.float32), channels_dim),
                                 max_outputs=1)
                    )

        return tf.summary.merge(images)

    def fetch_next_batch(self, mode='primary', metadata=False):

        minibatch_data = self._larcv_interface.fetch_minibatch_data(mode, fetch_meta_data=metadata)
        minibatch_dims = self._larcv_interface.fetch_minibatch_dims(mode)


        for key in minibatch_data:
            if key == 'entries' or key == 'event_ids':
                continue
            minibatch_data[key] = numpy.reshape(minibatch_data[key], minibatch_dims[key])

        minibatch_data['image']  = data_transforms.larcvsparse_to_dense_2d(minibatch_data['image'], dense_shape=FLAGS.SHAPE)
        minibatch_data['label']  = data_transforms.larcvsparse_to_dense_2d(minibatch_data['label'], dense_shape=FLAGS.SHAPE)

        minibatch_data['weight'] = self.compute_weights(minibatch_data['label'])

        return minibatch_data


    def compute_weights(self, labels, boost_labels = None):
        '''
        This is NOT a tensorflow implementation, but a numpy implementation.
        Running on CPUs this might not make a difference.  Running on GPUs
        it might be good to move this to a GPU, but I suspect it's not needed.
        '''
        # Take the labels, and compute the per-label weight


        # Prepare output weights:
        weights = numpy.zeros(labels.shape)

        i = 0
        for batch in labels:
            # First, figure out what the labels are and how many of each:
            values, counts = numpy.unique(batch, return_counts=True)

            n_pixels = numpy.sum(counts)
            for value, count in zip(values, counts):
                weight = 1.0*(n_pixels - count) / n_pixels
                if boost_labels is not None and value in boost_labels.keys():
                    weight *= boost_labels[value]
                mask = labels[i] == value
                weights[i, mask] += weight
            weights[i] *= 1. / numpy.sum(weights[i])
            i += 1

        # Normalize the weights to sum to 1 for each event:
        return weights

    def increment_global_step(self):
        raise NotImplementedError("You must implement this function")

    def on_step_end(self):
        pass

    def on_epoch_end(self):
        pass

    def val_step(self, gs):

        if gs == 0: return

        if self._val_writer is None:
            return

        if gs % FLAGS.AUX_ITERATION == 0:
            start = time.time()
            minibatch_data = self.fetch_next_batch('aux')
            io_end = time.time()

            minibatch_data['io_time'] = io_end - start
            print("Fetched batch")
            if gs % 10*FLAGS.AUX_ITERATION == 0:
                print ("Running images")
                val_images, val_summary = self._sess.run([self._summary_images, self._summary_basic],
                    feed_dict = self.feed_dict(inputs=minibatch_data))
                self._val_writer.add_summary(val_images, gs)
            else:
                print ("Running BASIC")
                val_summary = self._sess.run(self._summary_basic, 
                    feed_dict = self.feed_dict(inputs=minibatch_data))
            print("Ran validation")
            self._val_writer.add_summary(val_summary, gs)

        return


    def train_step(self):


        start = time.time()

        minibatch_data = self.fetch_next_batch()

        io_end = time.time()

        # print("Got data from file")

        # Reshape labels by dict entry, if needed, or all at once:
        # minibatch_data['label'] = numpy.reshape(minibatch_data['label'], minibatch_dims['label'])

        minibatch_data['io_time'] = io_end - start

        fd = self.feed_dict(minibatch_data)

        gs, _  = self._sess.run([self._global_step, self._train_op], 
                       feed_dict = self.feed_dict(inputs = minibatch_data))

        if not FLAGS.DISTRIBUTED:
            self._larcv_interface.next('primary')

        return gs

 
    def stop(self):
        # Mostly, this is just turning off the io:
        self._larcv_interface.stop()

    def checkpoint(self):

        if self._global_step % FLAGS.CHECKPOINT_ITERATION == 0 and self._global_step != 0:
            # Save a checkpoint, but don't do it on the first pass
            self.save_model()


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
            if inputs[key] is not None:
                fd.update({self._input[key] : inputs[key]})

        return fd


    def batch_process(self, verbose=True):

        # Run iterations
        for i in range(FLAGS.ITERATIONS):
            if FLAGS.TRAINING and self._iteration >= FLAGS.ITERATIONS:
                print('Finished training (iteration %d)' % self._iteration)
                break

            if verbose:
                try:
                    print ("Step {}, Current images per second: ".format(i), FLAGS.MINIBATCH_SIZE / (time.time() - self._snapshot))
                except:
                    pass
                    
                self._snapshot = time.time()

            gs = self.train_step()
            self.val_step(gs)

