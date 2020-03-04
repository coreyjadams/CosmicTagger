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
        # sys.stdout.write("Begin constructing network\n")



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

        self._metrics = {}

        if self.args.conv_mode == '2D':
            self._net = uresnet2D.UResNet(self.args)
        else:
            self._net = uresnet3D.UResNet3D(self.args)

        self._net.trainable = True

        self._logits = self._net(self._input['image'], training=self.args.training)


        # Used to accumulate gradients over several iterations:
        with tf.variable_scope("gradient_accumulation"):
            self._accum_vars = [tf.Variable(tv.initialized_value(),
                                trainable=False) for tv in tf.trainable_variables()]


        if self.args.mode == "train" or self.args.mode == "inference":


            # Here, if the data format is channels_first, we have to reorder the logits tensors
            # To put channels last.  Otherwise it does not work with the softmax tensors.

            # if self.args.data_format != "channels_last":
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
            with tf.variable_scope("prediction"):
                # self._output['softmax'] = [ tf.nn.softmax(x) for x in self._logits]
                self._output['prediction'] = [ tf.argmax(x, axis=self._channels_dim) for x in self._logits]

            with tf.variable_scope("cross_entropy"):
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
                        logits = self._logits)


            self._accuracy_calc = AccuracyCalculator.AccuracyCalculator()



            self._accuracy = self._accuracy_calc(prediction=self._output['prediction'], labels=self._input['split_labels'])

            # Add the metrics by hand:

            self._metrics = {}
            for p in [0,1,2]:
                self._metrics[f"plane{p}/Total_Accuracy"]          = self._accuracy["total_accuracy"][p]
                self._metrics[f"plane{p}/Non_Background_Accuracy"] = self._accuracy["non_bkg_accuracy"][p]
                self._metrics[f"plane{p}/Neutrino_IoU"]            = self._accuracy["neut_iou"][p]
                self._metrics[f"plane{p}/Cosmic_IoU"]              = self._accuracy["cosmic_iou"][p]

            with tf.variable_scope("accuracy"):
                self._metrics["Average/Total_Accuracy"]          = tf.reduce_mean(self._accuracy["total_accuracy"])
                self._metrics["Average/Non_Background_Accuracy"] = tf.reduce_mean(self._accuracy["non_bkg_accuracy"])
                self._metrics["Average/Neutrino_IoU"]            = tf.reduce_mean(self._accuracy["neut_iou"])
                self._metrics["Average/Cosmic_IoU"]              = tf.reduce_mean(self._accuracy["cosmic_iou"])




            self.loss_calculator = LossCalculator.LossCalculator(self.args.loss_balance_scheme, self._channels_dim)


            self._metrics['loss'] = self._loss




        self._log_keys = ["loss", "Average/Non_Background_Accuracy"]

        end = time.time()
        return end - start

    def print_network_info(self):
        n_trainable_parameters = 0
        for var in tf.trainable_variables():
            n_trainable_parameters += numpy.prod(var.get_shape())
            # print(var.name, var.get_shape())
        sys.stdout.write("Total number of trainable parameters in this network: {}\n".format(n_trainable_parameters))


    def set_compute_parameters(self):

        self._config = tf.compat.v1.ConfigProto()

        if self.args.compute_mode == "CPU":
            self._config.inter_op_parallelism_threads = self.args.inter_op_parallelism_threads
            self._config.intra_op_parallelism_threads = self.args.intra_op_parallelism_threads
        if self.args.compute_mode == "GPU":
            self._config.gpu_options.allow_growth = True
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "True"

    def initialize(self, io_only=False):


        self._initialize_io(color=self._rank)

        self.init_global_step()

        if io_only:
            return


        start = time.time()
        graph = tf.compat.v1.get_default_graph()
        net_time = self.init_network()

        sys.stdout.write("Done constructing network. ({0:.2}s)\n".format(time.time()-start))


        self.print_network_info()

        if self.args.mode != "inference":
            self.init_optimizer()

        self.init_saver()

        # Take all of the metrics and turn them into summaries:
        for key in self._metrics:
            tf.compat.v1.summary.scalar(key, self._metrics[key])

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
        self._learning_rate = self.args.learning_rate


    def restore_model(self):
        ''' This function attempts to restore the model from file
        '''

        if self.args.checkpoint_directory == None:
            file_path= self.args.log_directory  + "/checkpoints/"
        else:
            file_path= self.args.checkpoint_directory  + "/checkpoints/"

        path = tf.train.latest_checkpoint(file_path)


        if path is None:
            self.print("No checkpoint found, starting from scratch")
            return False
        # Parse the checkpoint file and use that to get the latest file path
        self.print("Restoring checkpoint from ", path)
        self._saver.restore(self._sess, path)

        return True

    def checkpoint(self, global_step):

        if global_step % self.args.checkpoint_iteration == 0 and global_step != 0:
            # Save a checkpoint, but don't do it on the first pass
            self.save_model(global_step)


    def save_model(self, global_step):
        '''Save the model to file

        '''

        # name, checkpoint_file_path = self.get_model_filepath(global_step)
        # Find the base path of the log directory
        if self.args.checkpoint_directory == None:
            file_path= self.args.log_directory  + "/checkpoints/"
        else:
            file_path= self.args.checkpoint_directory  + "/checkpoints/"


        # # Make sure the path actually exists:
        # if not os.path.isdir(os.path.dirname(file_path)):
        #     os.makedirs(os.path.dirname(file_path))

        saved_path = self._saver.save(self._sess, file_path + "model_{}.ckpt".format(global_step))


    def get_model_filepath(self, global_step):
        '''Helper function to build the filepath of a model for saving and restoring:

        '''

        # Find the base path of the log directory
        if self.args.checkpoint_directory == None:
            file_path= self.args.log_directory  + "/checkpoints/"
        else:
            file_path= self.args.checkpoint_directory  + "/checkpoints/"


        name = file_path + 'model-{}.ckpt'.format(global_step)
        checkpoint_file_path = file_path + "checkpoint"

        return name, checkpoint_file_path


    def init_saver(self):

        if self.args.checkpoint_directory == None:
            file_path= self.args.log_directory  + "/checkpoints/"
        else:
            file_path= self.args.checkpoint_directory  + "/checkpoints/"

        try:
            os.makedirs(file_path)
        except:
            self.print("Could not make file path")

        # Create a saver for snapshots of the network:
        self._saver = tf.compat.v1.train.Saver()
        self._saver_dir = file_path

        # Create a file writer for training metrics:
        self._main_writer = tf.compat.v1.summary.FileWriter(logdir=self.args.log_directory+"/train/")

        # Additionally, in training mode if there is aux data use it for validation:
        if self.args.aux_file is not None:
            self._val_writer = tf.compat.v1.summary.FileWriter(logdir=self.args.log_directory+"/test/")

    def init_global_step(self):
        self._global_step = tf.compat.v1.train.get_or_create_global_step()


    def init_optimizer(self):

        self.init_learning_rate()


        if 'RMS' in self.args.optimizer.upper():
            # Use RMS prop:
            self.print("Selected optimizer is RMS Prop")
            self._opt = tf.compat.v1.train.RMSPropOptimizer(self._learning_rate)
        elif 'LARS' in self.args.optimizer.upper():
            self.print("Selected optimizer is LARS")
            self._opt = tf.contrib.opt.LARSOptimizer(self._learning_rate)
        else:
            # default is Adam:
            self.print("Using default Adam optimizer")
            self._opt = tf.compat.v1.train.AdamOptimizer(self._learning_rate)



        # self._train_op = self._opt.minimize(self._loss, self._global_step)

        with tf.name_scope('gradient_accumulation'):

            self._zero_gradients =  [tv.assign(tf.zeros_like(tv)) for tv in self._accum_vars]
            self._accum_gradients = [self._accum_vars[i].assign_add(gv[0]) for
                                     i, gv in enumerate(self._opt.compute_gradients(self._loss))]

            self._apply_gradients = self._opt.apply_gradients(zip(self._accum_vars, tf.trainable_variables()),
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

        with tf.compat.v1.variable_scope('summary_images/'):


            images = []

            # Labels is an unsplit tensor, prediction is a split tensor
            split_labels = [ tf.cast(l, floating_point_format) for l in tf.split(labels,len(prediction) , self._channels_dim)]
            prediction = [ tf.expand_dims(tf.cast(p, floating_point_format), self._channels_dim) for p in prediction ]

            if self.args.data_format == "channels_first":
                split_labels = [ tf.transpose(l, [0, 2, 3, 1]) for l in split_labels]
                prediction   = [ tf.transpose(p, [0, 2, 3, 1]) for p in prediction]


            for p in range(len(split_labels)):

                images.append(
                    tf.compat.v1.summary.image('label_plane_{}'.format(p),
                                 split_labels[p],
                                 max_outputs=1)
                    )
                images.append(
                    tf.compat.v1.summary.image('pred_plane_{}'.format(p),
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

    def val_step(self, gs):


        if self._val_writer is None or self.args.synthetic:
            return

        if gs % self.args.aux_iteration == 0:


            do_summary_images = self._iteration != 0 and self._iteration % 50*self.args.summary_iteration == 0

            if self.args.no_summary_images:
                do_summary_images = False

            # Fetch the next batch of data with larcv
            minibatch_data = self.larcv_fetcher.fetch_next_batch('aux',force_pop=True)

            # For tensorflow, we have to build up an ops list to submit to the
            # session to run.

            # These are ops that always run:
            ops = {}
            ops['global_step'] = self._global_step
            ops['summary'] = self._summary_basic

            if do_summary_images:
                ops["summary_images"] = self._summary_images

            ops['metrics'] = self._metrics


            ops = self._sess.run(ops, feed_dict = self.feed_dict(inputs = minibatch_data))

            metrics = self.metrics(ops["metrics"])

            verbose = False




            if verbose: self.print("Calculated metrics")

            # Report metrics on the terminal:
            self.log(ops["metrics"], kind="Test", step=ops["global_step"])


            if verbose: self.print("Completed Log")

            self.write_summaries(self._val_writer, ops["summary"], ops["global_step"])


            if do_summary_images:
                self.write_summaries(self._val_writer, ops["summary_images"], ops["global_step"])


            if verbose: self.print("Summarized")



            return ops["global_step"]
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


            ##############################################
            # This is for NOT profiling.
            ops = self._sess.run(ops, feed_dict = self.feed_dict(inputs = minibatch_data))
            ##############################################

            # ##############################################
            # # THis is all for profiling:
            # # setup for timeline profile
            # run_meta = tf.RunMetadata()
            # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            #
            #
            # ops = self._sess.run(ops, feed_dict = self.feed_dict(inputs = minibatch_data),
            #     options=run_options, run_metadata=run_meta)
            #
            # NAME = "mb_{}_step_{}".format(minibatch_data['image'].shape[0], self._iteration)
            # self._main_writer.add_run_metadata(run_meta, NAME , self._iteration)
            #
            # # dump profile
            # tl = timeline.Timeline(run_meta.step_stats)
            # ctf = tl.generate_chrome_trace_format()
            # # dump file per iteration
            # with open('timeline_%s.json' % self._iteration, 'w') as f:
            #     f.write(ctf)
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
        self._sess.run(self._apply_gradients)

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

        return ops["global_step"]





    def stop(self):
        # Mostly, this is just turning off the io:
        # self._larcv_interface.stop()
        pass


    def ana_step(self):


        global_start_time = datetime.datetime.now()

        # Fetch the next batch of data with larcv
        io_start_time = datetime.datetime.now()
        minibatch_data = self.larcv_fetcher.fetch_next_batch("aux", metadata=True)
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
        self.log(ops["metrics"], kind="Inference", step=ops["global_step"])

        self.print(ops["metrics"])


        # Here is the part where we have to add output:

        if self.args.aux_file is not None:

            if self.args.data_format == "channels_last":
                locs = [ numpy.where(minibatch_data['image'][0,:,:,i] != 0) for i in [0,1,2]]
            else:
                locs = [ numpy.where(minibatch_data['image'][0,i,:,:] != 0) for i in [0,1,2]]

            for i, label in zip([1,2], ['neutrino', 'cosmic']):
                softmax    = []
                prediction = []
                for plane in [0,1,2]:
                    if self.args.data_format == "channels_first":
                        softmax.append(ops['softmax'][plane][0,i,:,:])
                        # locs = numpy.where(ops['prediction'][plane][0,:,:]) == i
                        # prediction.append({
                        #         'index'  : locs,
                        #         'values' : ops['prediction'][plane][locs],
                        #         'shape'  : ops['prediction'][plane].shape
                        #         }
                        #     )
                    else:
                        softmax.append(ops['softmax'][plane][0,:,:,i])

                    shape = ops['prediction'][plane][0].shape
                    locs_flat = numpy.ravel_multi_index(
                        multi_index = locs[plane],
                        dims        = shape
                    )
                    prediction.append({
                            'index'  : locs_flat,
                            'values' : ops['softmax'][plane][0][locs[plane]],
                            'shape'  : shape
                            }
                        )

                # self._larcv_interface.write_output(data=softmax,
                #     datatype='image2d',
                #     producer="seg_{}".format(label),
                #     entries=minibatch_data['entries'],
                #     event_ids=minibatch_data['event_ids'])

                self._larcv_interface.write_output(
                    data=prediction,
                    datatype='sparse2d',
                    producer = 'seg_{}'.format(label),
                    entries=minibatch_data['entries'],
                    event_ids=minibatch_data['event_ids'])



        if verbose: self.print("Completed Log")

        global_end_time = datetime.datetime.now()

        # Compute global step per second:
        self._seconds_per_global_step = (global_end_time - global_start_time).total_seconds()
        self._global_step += 1

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
        return fd


    def batch_process(self, verbose=True):

        start = time.time()
        post_one_time = None
        post_two_time = None
        # Run iterations
        for self._iteration in range(self.args.iterations):
            if self.args.training and self._iteration >= self.args.iterations:
                self.print('Finished training (iteration %d)' % self._iteration)
                break

            if self.args.mode == 'train':
                gs = self.train_step()
                self.val_step(gs)
                self.checkpoint(gs)
            elif self.args.mode == 'inference':
                self.ana_step()
            else:
                raise Exception("Don't know what to do with mode ", self.args.mode)

            if post_one_time is None:
                post_one_time = time.time()
            elif post_two_time is None:
                post_two_time = time.time()

        if self.args.mode == 'inference':
            if self._larcv_interface._writer is not None:
                self._larcv_interface._writer.finalize()

        end = time.time()

        self.print("Total time to batch_process: ", end - start)
        self.print("Total time to batch process except first iteration: ", end - post_one_time)
        self.print("Total time to batch process except first two iterations: ", end - post_two_time)
