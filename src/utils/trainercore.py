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
            

        self._log_keys = ["cross_entropy/Total_Loss", "accuracy/All_Plane_Neutrino_Accuracy"]

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

        self.init_saver()

        # Take all of the metrics and turn them into summaries:
        for key in self._metrics:
            tf.summary.scalar(key, self._metrics[key])

        self._summary_basic = tf.summary.merge_all()
        self._summary_images = self._create_summary_images(self._input['label'], self._output['prediction'])


        self.set_compute_parameters()

        # hooks = self.get_standard_hooks()

        if FLAGS.CHECKPOINT_DIRECTORY is not None:
            checkpoint_dir = FLAGS.CHECKPOINT_DIRECTORY
        else:
            checkpoint_dir = FLAGS.LOG_DIRECTORY

        # Add the graph to the log file:
        self._main_writer.add_graph(graph)

        self._sess = tf.Session(config = self._config)

        self._sess.run(tf.global_variables_initializer())

        # # Create a session:
        # self._sess = tf.train.MonitoredTrainingSession(config=self._config, hooks = hooks,
        #     checkpoint_dir        = checkpoint_dir,
        #     log_step_count_steps  = FLAGS.LOGGING_ITERATION,
        #     save_checkpoint_steps = FLAGS.CHECKPOINT_ITERATION)





    def get_model_filepath(self):
        '''Helper function to build the filepath of a model for saving and restoring:
        
        '''

        # Find the base path of the log directory
        if FLAGS.CHECKPOINT_DIRECTORY == None:
            file_path= FLAGS.LOG_DIRECTORY  + "/checkpoints/"
        else:
            file_path= FLAGS.CHECKPOINT_DIRECTORY  + "/checkpoints/"


        name = file_path + 'model-{}.ckpt'.format(self._global_step)
        checkpoint_file_path = file_path + "checkpoint"

        return name, checkpoint_file_path


    def init_saver(self):

        if FLAGS.CHECKPOINT_DIRECTORY == None:
            file_path= FLAGS.LOG_DIRECTORY  + "/checkpoints/"
        else:
            file_path= FLAGS.CHECKPOINT_DIRECTORY  + "/checkpoints/"

        try:
            os.makedirs(file_path)
        except FileExistsError:
            pass
        except:
            tf.log.error("Could not make file path")

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
                    # print(loss[p].get_shape())
                    # print(split_weights[p].get_shape())
                    loss[p] = tf.multiply(loss[p], split_weights[p])
                    # print(loss[p].get_shape())
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
            neut_iou         = [None]*len(logits['prediction'])
            cosmic_iou       = [None]*len(logits['prediction'])
            
            split_labels = [tf.squeeze(l, axis=channels_dim) for l in tf.split(labels,len(logits['prediction']) , channels_dim)]

            for p in range(len(logits['prediction'])):

                total_accuracy[p] = tf.reduce_mean(
                        tf.cast(tf.equal(logits['prediction'][p], split_labels[p]), tf.float32)
                    )
                # Find the non zero split_labels:
                non_zero_indices = tf.not_equal(split_labels[p], tf.constant(0, split_labels[p].dtype))

                # Find the neutrino indices:
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

                neut_iou[p] = tf.reduce_sum(tf.cast(neutrino_intersection, tf.float32)) / \
                  tf.reduce_sum(tf.cast(neutrino_union, tf.float32))

                cosmic_intersection = tf.math.logical_and(predicted_cosmic_indices, cosmic_indices)
                cosmic_union = tf.math.logical_or(predicted_cosmic_indices, cosmic_indices)

                cosmic_iou[p] = tf.reduce_sum(tf.cast(cosmic_intersection, tf.float32)) / \
                  tf.reduce_sum(tf.cast(cosmic_union, tf.float32))


                non_bkg_accuracy[p] = tf.reduce_mean(tf.cast(tf.equal(non_zero_logits, non_zero_labels), tf.float32))
                neut_accuracy[p]    = tf.reduce_mean(tf.cast(tf.equal(neutrino_logits, neutrino_labels), tf.float32))

                # Add the accuracies to the summary:
                self._metrics["split_accuracy/plane{0}/Total_Accuracy".format(p)] = total_accuracy[p]
                self._metrics["split_accuracy/plane{0}/Non_Background_Accuracy".format(p)] = non_bkg_accuracy[p]
                self._metrics["split_accuracy/plane{0}/Neutrino_Accuracy".format(p)] = neut_accuracy[p]
                self._metrics["split_accuracy/plane{0}/Neutrino_IoU".format(p)] = neut_iou[p]
                self._metrics["split_accuracy/plane{0}/Cosmic_IoU".format(p)] = neut_iou[p]

            #Compute the total accuracy and non background accuracy for all planes:
            all_accuracy            = tf.reduce_mean(total_accuracy)
            all_non_bkg_accuracy    = tf.reduce_mean(non_bkg_accuracy)
            all_neut_accuracy       = tf.reduce_mean(neut_accuracy)
            all_neut_iou            = tf.reduce_mean(neut_iou)
            all_cosmic_iou          = tf.reduce_mean(neut_iou)

            # Add the accuracies to the summary:
            self._metrics["accuracy/All_Plane_Total_Accuracy"] = all_accuracy
            self._metrics["accuracy/All_Plane_Non_Background_Accuracy"] = all_non_bkg_accuracy
            self._metrics["accuracy/All_Plane_Neutrino_Accuracy"] = all_neut_accuracy
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

    def on_step_end(self):
        pass

    def on_epoch_end(self):
        pass

    def val_step(self, gs):

        if gs == 0: return

        if self._val_writer is None:
            return

        if gs % FLAGS.AUX_ITERATION == 0:

            global_start_time = datetime.datetime.now()

            # Fetch the next batch of data with larcv
            io_start_time = datetime.datetime.now()
            minibatch_data = self.fetch_next_batch()
            io_end_time = datetime.datetime.now()

            # For tensorflow, we have to build up an ops list to submit to the
            # session to run.

            # These are ops that always run:
            ops = {}
            ops['global_step'] = self._global_step
            ops['summary'] = self._summary_basic

            ops['metrics'] = self._metrics

            if self._iteration != 0 and self._iteration % 25*FLAGS.SUMMARY_ITERATION == 0:
                ops['summary_images'] = self._summary_images


            ops = self._sess.run(ops, feed_dict = self.feed_dict(inputs = minibatch_data))

            metrics = ops["metrics"]

            verbose = False




            metrics['io_fetch_time'] = (io_end_time - io_start_time).total_seconds()

            if verbose: print("Calculated metrics")

            # Report metrics on the terminal:
            self.log(ops["metrics"], kind="Test", step=ops["global_step"]) 


            if verbose: print("Completed Log")


            self._aux_writer.add_summary(ops["summary"], global_step = ops["global_step"])
            if self._iteration != 0 and self._iteration % 25*FLAGS.SUMMARY_ITERATION == 0:
                self._aux_writer.add_summary(ops["summary"], global_step = ops["global_step"])


            # Create some extra summary information:
            extra_summary = tf.Summary(
                value=[
                    tf.Summary.Value(tag="io_fetch_time", simple_value=metrics['io_fetch_time']),
                ])

            self._aux_writer.add_summary(extra_summary, global_step=ops["global_step"])

            if verbose: print("Summarized")

            global_end_time = datetime.datetime.now()

            # Compute global step per second:
            self._seconds_per_global_step = (global_end_time - global_start_time).total_seconds()

            # Lastly, call next on the IO:
            self._larcv_interface.next('primary')

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

        if self._iteration != 0 and self._iteration % 25*FLAGS.SUMMARY_ITERATION == 0:
            ops['summary_images'] = self._summary_images


        ops = self._sess.run(ops, feed_dict = self.feed_dict(inputs = minibatch_data))

        metrics = ops["metrics"]

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


        self._main_writer.add_summary(ops["summary"], global_step = ops["global_step"])
        if self._iteration != 0 and self._iteration % 25*FLAGS.SUMMARY_ITERATION == 0:
            self._main_writer.add_summary(ops["summary_images"], global_step = ops["global_step"])


        # Create some extra summary information:
        extra_summary = tf.Summary(
            value=[
                tf.Summary.Value(tag="io_fetch_time", simple_value=metrics['io_fetch_time']),
                tf.Summary.Value(tag="global_step_per_sec", simple_value=metrics['global_step_per_sec']),
                tf.Summary.Value(tag="images_per_second", simple_value=metrics['images_per_second']),
            ])

        self._main_writer.add_summary(extra_summary, global_step=ops["global_step"])

        if verbose: print("Summarized")

        global_end_time = datetime.datetime.now()

        # Compute global step per second:
        self._seconds_per_global_step = (global_end_time - global_start_time).total_seconds()

        # Lastly, call next on the IO:
        self._larcv_interface.next('primary')

        return ops["global_step"]




 
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
        for self._iteration in range(FLAGS.ITERATIONS):
            if FLAGS.TRAINING and self._iteration >= FLAGS.ITERATIONS:
                print('Finished training (iteration %d)' % self._iteration)
                break

            gs = self.train_step()
            self.val_step(gs)

