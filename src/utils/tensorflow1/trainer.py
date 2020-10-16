import os
import sys
import time
import tempfile
from collections import OrderedDict

import numpy


from src.utils.core.trainercore import trainercore
from src.networks.tensorflow    import uresnet2D, uresnet3D, LossCalculator, AccuracyCalculator


import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'


import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow.python.client import timeline

floating_point_format = tf.float32
integer_format = tf.int64



class tf_trainer(trainercore):
    '''
    This is the tensorflow version of the trainer

    '''

    def __init__(self, args):
        trainercore.__init__(self, args)
        self._rank = None

    def local_batch_size(self):
        return self.args.minibatch_size

    def init_network(self):

        # This function builds the compute graph.
        # Optionally, it can build a 'subset' graph if this mode is

        # Net construction:
        start = time.time()

        # Here, if using mixed precision, set a global policy:
        if self.args.precision == "mixed":
            from tensorflow.keras.mixed_precision import experimental as mixed_precision
            self.policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(self.policy)

        batch_dims = self.larcv_fetcher.batch_dims(1)

        # We compute the
        batch_dims[0] = self.local_batch_size()

        # We have to make placeholders for input objects:

        self._input = {
            'image'   : tf.compat.v1.placeholder(floating_point_format, batch_dims, name="input_image"),
            'label'   : tf.compat.v1.placeholder(integer_format,        batch_dims, name="input_label"),
            'io_time' : tf.compat.v1.placeholder(floating_point_format, (), name="io_fetch_time")
        }

        # Build the network object, forward pass only:

        if self.args.conv_mode == '2D':
            self._net = uresnet2D.UResNet(self.args)
        else:
            self._net = uresnet3D.UResNet3D(self.args)

        self._net.trainable = True

        self._logits = self._net(self._input['image'], training=self.args.training)

        # If channels first, need to permute the logits:
        if self._channels_dim == 1:
            permutation = tf.keras.layers.Permute((2, 3, 1))
            self._loss_logits = [ permutation(l) for l in self._logits ]
        else:
            self._loss_logits = self._logits


        # Used to accumulate gradients over several iterations:
        with tf.compat.v1.variable_scope("gradient_accumulation"):
            self._accum_vars = [tf.Variable(tv.initialized_value(),
                                trainable=False) for tv in tf.compat.v1.trainable_variables()]



        if self.args.mode == "train" or self.args.mode == "inference":


            # Here, if the data format is channels_first, we have to reorder the logits tensors
            # To put channels last.  Otherwise it does not work with the softmax tensors.


            # Apply a softmax and argmax:
            self._output = dict()

            # Take the logits (which are one per plane) and create a softmax and prediction (one per plane)
            with tf.compat.v1.variable_scope("prediction"):
                self._output['prediction'] = [ tf.argmax(x, axis=self._channels_dim) for x in self._logits]

            with tf.compat.v1.variable_scope("cross_entropy"):
                self.loss_calculator = LossCalculator.LossCalculator(self.args.loss_balance_scheme, self._channels_dim)

                self._input['split_labels'] = [
                    tf.squeeze(l, axis=self._channels_dim)
                        for l in tf.split(self._input['label'], 3, self._channels_dim)
                    ]
                self._input['split_images'] = [
                    tf.squeeze(l, axis=self._channels_dim)
                        for l in tf.split(self._input['image'], 3, self._channels_dim)
                    ]

                self._loss = self.loss_calculator(
                        labels = self._input['split_labels'],
                        logits = self._loss_logits)


            if self.args.mode == "inference":
                self._output['softmax'] = [tf.nn.softmax(x, axis=self._channels_dim) for x in self._logits]


            self._accuracy_calc = AccuracyCalculator.AccuracyCalculator()



            self._accuracy = self._accuracy_calc(prediction=self._output['prediction'], labels=self._input['split_labels'])

            # Add the metrics by hand:

            self._metrics = {}
            for p in [0,1,2]:
                self._metrics[f"plane{p}/Total_Accuracy"]   = self._accuracy["total_accuracy"][p]
                self._metrics[f"plane{p}/Non_Bkg_Accuracy"] = self._accuracy["non_bkg_accuracy"][p]
                self._metrics[f"plane{p}/Neutrino_IoU"]     = self._accuracy["neut_iou"][p]
                self._metrics[f"plane{p}/Cosmic_IoU"]       = self._accuracy["cosmic_iou"][p]
                self._metrics[f"plane{p}/mIoU"]             = self._accuracy["miou"][p]

            with tf.compat.v1.variable_scope("accuracy"):
                self._metrics["Average/Total_Accuracy"]   = tf.reduce_mean(self._accuracy["total_accuracy"])
                self._metrics["Average/Non_Bkg_Accuracy"] = tf.reduce_mean(self._accuracy["non_bkg_accuracy"])
                self._metrics["Average/Neutrino_IoU"]     = tf.reduce_mean(self._accuracy["neut_iou"])
                self._metrics["Average/Cosmic_IoU"]       = tf.reduce_mean(self._accuracy["cosmic_iou"])
                self._metrics["Average/mIoU"]             = tf.reduce_mean(self._accuracy["miou"])


            self._metrics['loss'] = self._loss

        self._log_keys = ["loss", "Average/Non_Bkg_Accuracy", "Average/mIoU"]

        end = time.time()
        return end - start

    def print_network_info(self, verbose=False):
        if verbose:
            for var in tf.compat.v1.trainable_variables():
                print(var)

        self.print("Total number of trainable parameters in this network: {}\n".format(self.n_parameters()))


    def n_parameters(self):
        n_trainable_parameters = 0
        for var in tf.compat.v1.trainable_variables():
            n_trainable_parameters += numpy.prod(var.get_shape())

        return n_trainable_parameters

    def set_compute_parameters(self):

        self._config = tf.compat.v1.ConfigProto()

        if self.args.compute_mode == "CPU":
            self._config.inter_op_parallelism_threads = self.args.inter_op_parallelism_threads
            self._config.intra_op_parallelism_threads = self.args.intra_op_parallelism_threads
        elif self.args.compute_mode == "GPU":
            gpus = tf.config.experimental.list_physical_devices('GPU')

            # The code below is for MPS mode.  It is a bit of a hard-coded
            # hack.  Use with caution since the memory limit is set by hand.
            ####################################################################
            # print(gpus)
            # if gpus:
            #   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
            #   try:
            #     tf.config.experimental.set_virtual_device_configuration(
            #         gpus[0],
            #         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15000)])
            #     # tf.config.experimental.set_memory_growth(gpus[0], True)
            #     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            #     # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            #   except RuntimeError as e:
            #     # Virtual devices must be set before GPUs have been initialized
            #     print(e)
            ####################################################################


    def initialize(self, io_only=False):


        self._initialize_io(color=0)

        self.init_global_step()

        if io_only:
            return

        if self.args.training:
            self.build_lr_schedule()

        start = time.time()
        graph = tf.compat.v1.get_default_graph()
        net_time = self.init_network()

        self.print("Done constructing network. ({0:.2}s)\n".format(time.time()-start))


        self.print_network_info()

        if self.args.mode != "inference":
            self.init_optimizer()

        self.init_saver()

        # Take all of the metrics and turn them into summaries:
        for key in self._metrics:
            tf.compat.v1.summary.scalar(key, self._metrics[key])

        if self.args.training:
            # Add the learning rate as a summary too:
            tf.compat.v1.summary.scalar('learning_rate', self._learning_rate)

        if self.args.mode != "inference":

            self._summary_basic  = tf.compat.v1.summary.merge_all()
            self._summary_images = self._create_summary_images(self._input['label'], self._output['prediction'])
            # self.create_model_summaries()

        self.set_compute_parameters()

        self.write_graph_to_tensorboard(graph)

        self._sess = tf.compat.v1.Session(config = self._config)

        # Try to restore a model?
        restored = self.restore_model()

        if not restored:
            self._sess.run(tf.compat.v1.global_variables_initializer())

        # # Create a session:
        # self._sess = tf.train.MonitoredTrainingSession(config=self._config, hooks = hooks,
        #     checkpoint_dir        = checkpoint_dir,
        #     log_step_count_steps  = self.args.logging_iteration,
        #     save_checkpoint_steps = self.args.checkpoint_iteration)

    def write_graph_to_tensorboard(self, graph):
        # Add the graph to the log file:
        self._main_writer.add_graph(graph)


    def init_learning_rate(self):
        # Use a place holder for the learning rate :
        self._learning_rate = tf.compat.v1.placeholder(floating_point_format, (), name="lr")



    def restore_model(self):
        ''' This function attempts to restore the model from file
        '''

        file_path = self.get_checkpoint_dir()

        path = tf.train.latest_checkpoint(file_path)


        if path is None:
            self.print("No checkpoint found, starting from scratch")
            return False
        # Parse the checkpoint file and use that to get the latest file path
        self.print("Restoring checkpoint from ", path)
        self._saver.restore(self._sess, path)

        # self.scheduler.set_current_step(self.get_current_global_step())

        return True

    def checkpoint(self):

        gs = self.get_current_global_step()

        if gs % self.args.checkpoint_iteration == 0 and gs != 0:
            # Save a checkpoint, but don't do it on the first pass
            self.save_model(gs)

    def get_checkpoint_dir(self):

        # Find the base path of the log directory
        if self.args.checkpoint_directory == None:
            file_path= self.args.log_directory  + "/tf/checkpoints/"
        else:
            file_path= self.args.checkpoint_directory  + "/tf/checkpoints/"

        return file_path

    def save_model(self, global_step):
        '''Save the model to file

        '''

        file_path = self.get_checkpoint_dir()

        # # Make sure the path actually exists:
        # if not os.path.isdir(os.path.dirname(file_path)):
        #     os.makedirs(os.path.dirname(file_path))

        saved_path = self._saver.save(self._sess, file_path + "model_{}.ckpt".format(global_step))


    def get_model_filepath(self, global_step):
        '''Helper function to build the filepath of a model for saving and restoring:

        '''

        file_path = self.get_checkpoint_dir()

        name = file_path + 'model-{}.ckpt'.format(global_step)
        checkpoint_file_path = file_path + "checkpoint"

        return name, checkpoint_file_path


    def init_saver(self):

        file_path = self.get_checkpoint_dir()

        try:
            os.makedirs(file_path)
        except:
            self.print("Could not make file path")

        # Create a saver for snapshots of the network:
        self._saver = tf.compat.v1.train.Saver()

        # Create a file writer for training metrics:
        self._main_writer = tf.compat.v1.summary.FileWriter(logdir=self.args.log_directory+"/tf/train/")

        # Additionally, in training mode if there is aux data use it for validation:
        if self.args.aux_file is not None:
            self._val_writer = tf.compat.v1.summary.FileWriter(logdir=self.args.log_directory+"/tf/test/")

    def init_global_step(self):
        self._global_step = tf.compat.v1.train.get_or_create_global_step()


    def init_optimizer(self):

        self.init_learning_rate()


        if 'RMS' in self.args.optimizer.upper():
            # Use RMS prop:
            self.print("Selected optimizer is RMS Prop")
            self._opt = tf.compat.v1.train.RMSPropOptimizer(self._learning_rate)
        else:
            # default is Adam:
            self.print("Using default Adam optimizer")
            self._opt = tf.compat.v1.train.AdamOptimizer(self._learning_rate)

        if self.args.precision == "mixed":
            from tensorflow.keras.mixed_precision import experimental as mixed_precision
            self._opt = mixed_precision.LossScaleOptimizer(self._opt, loss_scale='dynamic')



        else:
            with tf.name_scope('gradient_accumulation'):

                self._zero_gradients =  [tv.assign(tf.zeros_like(tv)) for tv in self._accum_vars]
                self._accum_gradients = [self._accum_vars[i].assign_add(gv[0]) for
                                         i, gv in enumerate(self._opt.compute_gradients(self._loss))]
                self._apply_gradients = self._opt.apply_gradients(zip(self._accum_vars, tf.compat.v1.trainable_variables()),
                    global_step = self._global_step)


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

        self.print(log_string)

        return


    def _create_summary_images(self, labels, prediction):
        ''' Create images of the labels and prediction to show training progress
        '''



        images = []

        # Labels is an unsplit tensor, prediction is a split tensor
        split_labels = [ tf.cast(l, floating_point_format) for l in tf.split(labels,len(prediction) , self._channels_dim)]
        prediction = [ tf.expand_dims(tf.cast(p, floating_point_format), self._channels_dim) for p in prediction ]

        if self.args.data_format == "channels_first":
            split_labels = [ tf.transpose(l, [0, 2, 3, 1]) for l in split_labels]
            prediction   = [ tf.transpose(p, [0, 2, 3, 1]) for p in prediction]


        for p in range(len(split_labels)):


                images.append(
                    tf.compat.v1.summary.image('label/plane_{}'.format(p),
                                 split_labels[p],
                                 max_outputs=1)
                    )

                images.append(
                    tf.compat.v1.summary.image('prediction/plane_{}'.format(p),
                                 prediction[p],
                                 max_outputs=1)
                    )

        return tf.compat.v1.summary.merge(images)


    def create_model_summaries(self):
        # # return
        # optimizer = tf.train.AdamOptimizer(..)
        # grads = optimizer.compute_gradients(loss)
        # weights = self._net.trainable_variables
        # weight_sum_op = _accum_gradients
        # with tf.variable_scope)
        hist = []

        self._accum_vars,
        for var, grad in zip(tf.compat.v1.trainable_variables(), self._accum_gradients):
            name = var.name.replace("/",".")
            hist.append(tf.summary.histogram(name, var))
            hist.append(tf.summary.histogram(name  + "/grad/", grad))
        # grad_summ_op = tf.summary.merge([tf.summary.histogram("%s-grad" % g[1].name, g[0]) for g in grads])
        # grad_vals = sess.run(fetches=grad_summ_op, feed_dict = feed_dict)
        self.model_summary = tf.summary.merge(hist)
        # self.model_summary = tf.compat.v1.summary.merge(hist)

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

    def val_step(self):

        if self.args.aux_file is None:
            return

        if self.args.synthetic:
            return

        gs = self.get_current_global_step()

        if gs % self.args.aux_iteration == 0:


            do_summary_images = self._iteration != 0 and self._iteration % 50*self.args.summary_iteration == 0

            if self.args.no_summary_images:
                do_summary_images = False

            # Fetch the next batch of data with larcv
            minibatch_data = self.larcv_fetcher.fetch_next_batch('aux',force_pop=True)

            # Return if we get none:
            if minibatch_data is None: return

            # For tensorflow, we have to build up an ops list to submit to the
            # session to run.

            # These are ops that always run:
            ops = {}
            ops['summary'] = self._summary_basic

            if do_summary_images:
                ops["summary_images"] = self._summary_images

            ops['metrics'] = self._metrics


            ops = self._sess.run(ops, feed_dict = self.feed_dict(inputs = minibatch_data))

            metrics = self.metrics(ops["metrics"])

            verbose = False




            if verbose: self.print("Calculated metrics")

            # Report metrics on the terminal:
            self.log(ops["metrics"], kind="Test", step=gs)


            if verbose: self.print("Completed Log")

            self.write_summaries(self._val_writer, ops["summary"], gs)


            if do_summary_images:
                self.write_summaries(self._val_writer, ops["summary_images"], gs)


            if verbose: self.print("Summarized")



            return gs
        return

    def train_step(self):

        global_start_time = datetime.datetime.now()

        # For tensorflow, we have to build up an ops list to submit to the
        # session to run.

        metrics = None

        # First, zero out the gradients:
        self._sess.run(self._zero_gradients)
        io_fetch_time = 0.0

        do_summary_images = self._iteration != 0 and self._iteration % 1*self.args.summary_iteration == 0

        if self.args.no_summary_images:
            do_summary_images = False

        for i in range(self.args.gradient_accumulation):

            # Fetch the next batch of data with larcv
            io_start_time = datetime.datetime.now()
            minibatch_data = self.larcv_fetcher.fetch_next_batch("train",force_pop=True)


            # Abort if we get "None"
            if minibatch_data is None: return

            io_end_time = datetime.datetime.now()
            io_fetch_time += (io_end_time - io_start_time).total_seconds()


            # These are ops that always run:
            ops = {}
            ops['_accum_gradients']  = self._accum_gradients
            ops['global_step']       = self._global_step
            ops['metrics']           = self._metrics

            # Run the summary only once:
            if i == 0:
                ops['summary']           = self._summary_basic
                # ops['graph_summary']     = self.model_summary
                # Add the images, but only once:
                if do_summary_images:
                    ops['summary_images'] = self._summary_images


            fd = self.feed_dict(inputs = minibatch_data)

            if self.args.profile:
                # Run this if not data-parallel, or rank ==0 when data-parallel
                if not self.args.distributed or self._rank == 0:
                    run_meta = tf.RunMetadata()
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)


                    ops = self._sess.run(ops, feed_dict = self.feed_dict(inputs = minibatch_data),
                        options=run_options, run_metadata=run_meta)

                    NAME = "mb_{}_step_{}".format(minibatch_data['image'].shape[0], self._iteration)
                    self._main_writer.add_run_metadata(run_meta, NAME , self._iteration)

                    # dump profile
                    tl = timeline.Timeline(run_meta.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    # dump file per iteration
                    with open('timeline_%s.json' % self._iteration, 'w') as f:
                        f.write(ctf)
            else:
                # ##############################################
                # # This is for NOT profiling.
                    ops = self._sess.run(ops, feed_dict = fd)
                # ##############################################

            if metrics is None:
                metrics = self.metrics(ops["metrics"])
            else:
                temp_metrics = self.metrics(ops["metrics"])
                for key in metrics:
                    metrics[key] += temp_metrics[key]

            # Grab the summaries if we need to write them:
            if i == 0:
                summaries = ops['summary']
                # graph_summary = ops['graph_summary']
                if do_summary_images:
                    summary_images = ops['summary_images']

        # Lastly, update the weights:

        self._sess.run(self._apply_gradients,
            feed_dict = {self._learning_rate : fd[self._learning_rate]})

        # Normalize the metrics:
        for key in metrics:
            metrics[key] /= self.args.gradient_accumulation

        verbose = False

        # Add the global step / second to the tensorboard log:
        try:
            metrics['global_step_per_sec'] = 1./self._seconds_per_global_step
            metrics['images_per_second'] = (self.args.minibatch_size*self.args.gradient_accumulation) / self._seconds_per_global_step
        except AttributeError:
            metrics['global_step_per_sec'] = 0.0
            metrics['images_per_second'] = 0.0



        metrics['io_fetch_time'] = io_fetch_time

        if verbose: self.print("Calculated metrics")

        # Report metrics on the terminal:
        self.log(metrics, kind="Train", step=ops["global_step"])


        if verbose: self.print("Completed Log")

        # self.write_summaries(self._main_writer, graph_summary, ops["global_step"])


        self.write_summaries(self._main_writer, summaries, ops["global_step"])
        if do_summary_images:
            self.write_summaries(self._main_writer, summary_images, ops["global_step"])

        # Create some extra summary information:
        extra_summary = tf.compat.v1.Summary(
            value=[
                tf.compat.v1.Summary.Value(tag="io_fetch_time", simple_value=metrics['io_fetch_time']),
                tf.compat.v1.Summary.Value(tag="global_step_per_sec", simple_value=metrics['global_step_per_sec']),
                tf.compat.v1.Summary.Value(tag="images_per_second", simple_value=metrics['images_per_second']),
            ])

        self.write_summaries(self._main_writer, extra_summary, ops["global_step"])

        if verbose: self.print("Summarized")

        global_end_time = datetime.datetime.now()

        # Compute global step per second:
        self._seconds_per_global_step = (global_end_time - global_start_time).total_seconds()

        # self._global_step = ops["global_step"]

        return


    def get_current_global_step(self):
        return self._sess.run(self._global_step)


    def stop(self):
        # Mostly, this is just turning off the io:
        # self._larcv_interface.stop()
        pass


    def ana_step(self):


        global_start_time = datetime.datetime.now()

        # Fetch the next batch of data with larcv
        io_start_time = datetime.datetime.now()

        if self._iteration == 0:
            force_pop = False
        else:
            force_pop = True
        minibatch_data = self.larcv_fetcher.fetch_next_batch("train", force_pop=force_pop)


        # Escape if we get None:
        if minibatch_data is None: return

        io_end_time = datetime.datetime.now()

        # For tensorflow, we have to build up an ops list to submit to the
        # session to run.

        # These are ops that always run:
        ops = {}
        ops['logits']     = self._logits
        ops['softmax']    = self._output['softmax']
        ops['prediction'] = self._output['prediction']
        ops['metrics']    = self._metrics
        ops = self._sess.run(ops, feed_dict = self.feed_dict(inputs = minibatch_data))
        ops['global_step'] = self._global_step

        metrics = self.metrics(ops["metrics"])


        verbose = False

        # Add the global step / second to the tensorboard log:
        try:
            metrics['global_step_per_sec'] = 1./self._seconds_per_global_step
            metrics['images_per_second'] = self.args.minibatch_size / self._seconds_per_global_step
        except AttributeError:
            metrics['global_step_per_sec'] = 0.0
            metrics['images_per_second'] = 0.0



        metrics['io_fetch_time'] = (io_end_time - io_start_time).total_seconds()

        if verbose: self.print("Calculated metrics")

        # Report metrics on the terminal:
        self.log(ops["metrics"], kind="Inference", step=self.get_current_global_step())

        # self.print(ops["metrics"])


        # If there is an aux file, for ana that means an output file.
        # Call the larcv interface to write data:
        if self.args.aux_file is not None:

            # For writing output, we get the non-zero locations from the labels.
            # Then, we get the neutrino and cosmic scores for those locations in the logits,
            # After applying a softmax.

            # To do this, we just need to capture the following objects:
            # - the dense shape
            # - the location of all non-zero pixels, flattened
            # - the softmax score for all non-zero pixels, flattened.


            # Compute the softmax over all images in the batch:
            softmax = ops['softmax']
            for b_index in range(self.args.minibatch_size):

                # We want to make sure we get the locations from the non-zero input pixels:
                images = numpy.split(minibatch_data['image'][b_index,:,:,:], 3, axis=self._channels_dim)

                # Reshape images here to remove the empty index:
                images = [image.squeeze() for image in images]

                # Locations is a list of a tuple of coordinates for each image
                locations = [numpy.where(image != 0) for image in images]

                # Shape is a list of shapes for each image:
                shape = [ image.shape for image in images ]

                # Batch softmax is a list of the softmax tensor on each plane
                batch_softmax = [s[b_index] for s in softmax]


                # To get the neutrino scores, we want to access the softmax at the neutrino index
                # And slice over just the non-zero locations:
                if self._channels_dim == 1:
                    neutrino_scores = [ b[self.NEUTRINO_INDEX][locations[plane]]
                        for plane, b in enumerate(batch_softmax) ]
                    cosmic_scores   = [ b[self.COSMIC_INDEX][locations[plane]]
                        for plane, b in enumerate(batch_softmax) ]
                else:
                    neutrino_scores = [ b[locations[plane]][:,self.NEUTRINO_INDEX]
                        for plane, b in enumerate(batch_softmax) ]
                    cosmic_scores   = [ b[locations[plane]][:,self.COSMIC_INDEX]
                        for plane, b in enumerate(batch_softmax) ]

                # Lastly, flatten the locations.
                # For the unraveled index, there is a complication that torch stores images
                # with [H,W] and larcv3 stores images with [W, H] by default.
                # To solve this -
                # Reverse the shape:
                shape = [ s[::-1] for s in shape ]
                # Go through the locations in reverse:
                locations = [ numpy.ravel_multi_index(l[::-1], s) for l, s in zip(locations, shape) ]

                # Now, package up the objects to send to the file writer:
                neutrino_data = []
                cosmic_data = []
                for plane in range(3):
                    neutrino_data.append({
                        'values' : neutrino_scores[plane],
                        'index'  : locations[plane],
                        'shape'  : shape[plane]
                    })
                    cosmic_data.append({
                        'values' : cosmic_scores[plane],
                        'index'  : locations[plane],
                        'shape'  : shape[plane]
                    })


                # Write the data through the writer:
                self.larcv_fetcher.write(cosmic_data,
                    producer = "cosmic_prediction",
                    entry    = minibatch_data['entries'][b_index],
                    event_id = minibatch_data['event_ids'][b_index])

                self.larcv_fetcher.write(neutrino_data,
                    producer = "neutrino_prediction",
                    entry    = minibatch_data['entries'][b_index],
                    event_id = minibatch_data['event_ids'][b_index])


        if verbose: self.print("Completed Log")

        global_end_time = datetime.datetime.now()

        # Compute global step per second:
        self._seconds_per_global_step = (global_end_time - global_start_time).total_seconds()
        # self._global_step += 1

        return ops["global_step"]


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

        # fd[self._learning_rate] = self._base_learning_rate

        for key in inputs:
            if key == "entries" or key == "event_ids": continue

            if inputs[key] is not None:
                fd.update({self._input[key] : inputs[key]})

        if self.args.training:
            fd.update({self._learning_rate : self.lr_calculator(self.get_current_global_step())})
        return fd

    def close_savers(self):
        pass
        # if self.args.mode == 'inference':
        #     if self.larcv_fetcher._writer is not None:
        #         self.larcv_fetcher._writer.finalize()
