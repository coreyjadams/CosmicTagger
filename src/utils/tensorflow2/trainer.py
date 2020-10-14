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

        #
        self._global_step = tf.Variable(0, dtype=tf.int64)

        # We have to make placeholders for input objects:
        #
        # self._input = {
        #     'image'   : tf.compat.v1.placeholder(floating_point_format, batch_dims, name="input_image"),
        #     'label'   : tf.compat.v1.placeholder(integer_format,        batch_dims, name="input_label"),
        #     'io_time' : tf.compat.v1.placeholder(floating_point_format, (), name="io_fetch_time")
        # }

        # Build the network object, forward pass only:

        if self.args.conv_mode == '2D':
            self._net = uresnet2D.UResNet(self.args)
        else:
            self._net = uresnet3D.UResNet3D(self.args)

        self._net.trainable = True

        # self._logits = self._net(self._input['image'], training=self.args.training)

        # # If channels first, need to permute the logits:
        # if self._channels_dim == 1:
        #     permutation = tf.keras.layers.Permute((2, 3, 1))
        #     self._loss_logits = [ permutation(l) for l in self._logits ]
        # else:
        #     self._loss_logits = self._logits


        # TO PROPERLY INITIALIZE THE NETWORK, NEED TO DO A FORWARD PASS
        minibatch_data = self.larcv_fetcher.fetch_next_batch("train",force_pop=False)
        self.forward_pass(minibatch_data, training=False)


            # # Here, if the data format is channels_first, we have to reorder the logits tensors
            # # To put channels last.  Otherwise it does not work with the softmax tensors.
            #
            #
            # # Apply a softmax and argmax:
            # self._output = dict()
            #
            # # Take the logits (which are one per plane) and create a softmax and prediction (one per plane)
            # with tf.compat.v1.variable_scope("prediction"):
            #     self._output['prediction'] = [ tf.argmax(x, axis=self._channels_dim) for x in self._logits]
            #
            # with tf.compat.v1.variable_scope("cross_entropy"):
            #
            #     self._input['split_labels'] = [
            #         tf.squeeze(l, axis=self._channels_dim)
            #             for l in tf.split(self._input['label'], 3, self._channels_dim)
            #         ]
            #     self._input['split_images'] = [
            #         tf.squeeze(l, axis=self._channels_dim)
            #             for l in tf.split(self._input['image'], 3, self._channels_dim)
            #         ]
            #
            #     self._loss = self.loss_calculator(
            #             labels = self._input['split_labels'],
            #             logits = self._loss_logits)
            #
            #
            # if self.args.mode == "inference":
            #     self._output['softmax'] = [tf.nn.softmax(x, axis=self._channels_dim) for x in self._logits]


        self.acc_calculator  = AccuracyCalculator.AccuracyCalculator()
        self.loss_calculator = LossCalculator.LossCalculator(self.args.loss_balance_scheme, self._channels_dim)

        self._log_keys = ["loss", "Average/Non_Bkg_Accuracy", "Average/mIoU"]

        end = time.time()
        return end - start

    def print_network_info(self, verbose=False):
        n_trainable_parameters = 0
        for var in self._net.variables:
            n_trainable_parameters += numpy.prod(var.get_shape())
            if verbose:
                print(var.name, var.get_shape())
        self.print(f"Total number of trainable parameters in this network: {n_trainable_parameters}")


    def n_parameters(self):
        n_trainable_parameters = 0
        for var in tf.compat.v1.trainable_variables():
            n_trainable_parameters += numpy.prod(var.get_shape())

        return n_trainable_parameters

    def current_step(self):
        return self._global_step


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

    @tf.function
    def summary_images(self, labels, prediction):
        ''' Create images of the labels and prediction to show training progress
        '''

        if self.current_step() % 25 * self.args.summary_iteration == 0 and not self.args.no_summary_images:

            for p in range(len(labels)):
                tf.summary.image(f"label_plane_{p}", labels[p],     self.current_step())
                tf.summary.image(f"pred_plane_{p}",  prediction[p], self.current_step())

            # images = []

            # # Labels is an unsplit tensor, prediction is a split tensor
            # split_labels = [ tf.cast(l, floating_point_format) for l in tf.split(labels,len(prediction) , self._channels_dim)]
            # prediction = [ tf.expand_dims(tf.cast(p, floating_point_format), self._channels_dim) for p in prediction ]

            # if self.args.data_format == "channels_first":
            #     split_labels = [ tf.transpose(a=l, perm=[0, 2, 3, 1]) for l in split_labels]
            #     prediction   = [ tf.transpose(a=p, perm=[0, 2, 3, 1]) for p in prediction]


            # for p in range(len(split_labels)):

            #     images.append(
            #         tf.compat.v1.summary.image('label_plane_{}'.format(p),
            #                      split_labels[p],
            #                      max_outputs=1)
            #         )
            #     images.append(
            #         tf.compat.v1.summary.image('pred_plane_{}'.format(p),
            #                      prediction[p],
            #                      max_outputs=1)
            #         )

        return

    def initialize(self, io_only=False):


        self._initialize_io(color=0)


        if io_only:
            return

        if self.args.training:
            self.build_lr_schedule()

        start = time.time()

        net_time = self.init_network()

        self.print("Done constructing network. ({0:.2}s)\n".format(time.time()-start))


        self.print_network_info()

        if self.args.mode != "inference":
            self.init_optimizer()

        self.init_saver()
        #
        # # Take all of the metrics and turn them into summaries:
        # for key in self._metrics:
        #     tf.compat.v1.summary.scalar(key, self._metrics[key])
        #
        # if self.args.training:
        #     # Add the learning rate as a summary too:
        #     tf.compat.v1.summary.scalar('learning_rate', self._learning_rate)
        #
        # if self.args.mode != "inference":
        #
        #     self._summary_basic  = tf.compat.v1.summary.merge_all()
        #     self._summary_images = self._create_summary_images(self._input['label'], self._output['prediction'])
        #     # self.create_model_summaries()

        self.set_compute_parameters()


        # Try to restore a model?
        restored = self.restore_model()


    def init_learning_rate(self):
        # Use a place holder for the learning rate :
        self._learning_rate = tf.Variable(initial_value=0.0, trainable=False, dtype=floating_point_format, name="lr")



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
        self._net.load_weights(path)

        # self.scheduler.set_current_step(self.current_step())

        return True

    def checkpoint(self):

        gs = int(self.current_step().numpy())

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

        saved_path = self._net.save_weights(file_path + "model_{}.ckpt".format(global_step))


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

        # # Create a saver for snapshots of the network:
        # self._saver = tf.compat.v1.train.Saver()

        # Create a file writer for training metrics:
        self._main_writer = tf.summary.create_file_writer(self.args.log_directory+"/tf/train/")

        # Additionally, in training mode if there is aux data use it for validation:
        if self.args.aux_file is not None:
            self._val_writer = tf.summary.create_file_writer(self.args.log_directory+"/tf/test/")


    def init_optimizer(self):

        self.init_learning_rate()


        if 'RMS' in self.args.optimizer.upper():
            # Use RMS prop:
            self.print("Selected optimizer is RMS Prop")
            self._opt = tf.keras.optimizers.RMSprop(self._learning_rate)
        else:
            # default is Adam:
            self.print("Using default Adam optimizer")
            self._opt = tf.keras.optimizers.Adam(self._learning_rate)

        if self.args.precision == "mixed":
            from tensorflow.keras.mixed_precision import experimental as mixed_precision
            self._opt = mixed_precision.LossScaleOptimizer(self._opt, loss_scale='dynamic')

        self.tape = tf.GradientTape()

    def _compute_metrics(self, logits, prediction, labels, loss):

        # self._output['softmax'] = [ tf.nn.softmax(x) for x in self._logits]
        # self._output['prediction'] = [ tf.argmax(input=x, axis=self._channels_dim) for x in self._logits]
        accuracy = self.acc_calculator(prediction=prediction, labels=labels)

        metrics = {}
        for p in [0,1,2]:
            metrics[f"plane{p}/Total_Accuracy"]          = accuracy["total_accuracy"][p]
            metrics[f"plane{p}/Non_Bkg_Accuracy"]        = accuracy["non_bkg_accuracy"][p]
            metrics[f"plane{p}/Neutrino_IoU"]            = accuracy["neut_iou"][p]
            metrics[f"plane{p}/Cosmic_IoU"]              = accuracy["cosmic_iou"][p]

        metrics["Average/Total_Accuracy"]          = float(tf.reduce_mean(accuracy["total_accuracy"]).numpy())
        metrics["Average/Non_Bkg_Accuracy"]        = float(tf.reduce_mean(accuracy["non_bkg_accuracy"]).numpy())
        metrics["Average/Neutrino_IoU"]            = float(tf.reduce_mean(accuracy["neut_iou"]).numpy())
        metrics["Average/Cosmic_IoU"]              = float(tf.reduce_mean(accuracy["cosmic_iou"]).numpy())
        metrics["Average/mIoU"]                    = float(tf.reduce_mean(accuracy["miou"]).numpy())

        metrics['loss'] = loss

        return metrics

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

    #
    # def _create_summary_images(self, labels, prediction):
    #     ''' Create images of the labels and prediction to show training progress
    #     '''
    #
    #
    #
    #     images = []
    #
    #     # Labels is an unsplit tensor, prediction is a split tensor
    #     split_labels = [ tf.cast(l, floating_point_format) for l in tf.split(labels,len(prediction) , self._channels_dim)]
    #     prediction = [ tf.expand_dims(tf.cast(p, floating_point_format), self._channels_dim) for p in prediction ]
    #
    #     if self.args.data_format == "channels_first":
    #         split_labels = [ tf.transpose(l, [0, 2, 3, 1]) for l in split_labels]
    #         prediction   = [ tf.transpose(p, [0, 2, 3, 1]) for p in prediction]
    #
    #
    #     for p in range(len(split_labels)):
    #
    #
    #             images.append(
    #                 tf.compat.v1.summary.image('label/plane_{}'.format(p),
    #                              split_labels[p],
    #                              max_outputs=1)
    #                 )
    #
    #             images.append(
    #                 tf.compat.v1.summary.image('prediction/plane_{}'.format(p),
    #                              prediction[p],
    #                              max_outputs=1)
    #                 )
    #
    #     return tf.compat.v1.summary.merge(images)


    def forward_pass(self, minibatch_data, training):

        # Run a forward pass of the model on the input image:
        logits = self._net(minibatch_data['image'], training=training)
        labels = minibatch_data['label'].astype(numpy.int32)

        prediction = tf.argmax(logits, axis=self._channels_dim, output_type = tf.dtypes.int32)

        labels = tf.split(labels, num_or_size_splits=3, axis=self._channels_dim)
        labels = [tf.squeeze(li, axis=self._channels_dim) for li in labels]

        return labels, logits, prediction

    # @tf.function(experimental_relax_shapes=True)
    def summary(self, metrics,saver=""):

        if self.current_step() % self.args.summary_iteration == 0:

            if saver == "":
                saver = self._main_writer

            with saver.as_default():
                for metric in metrics:
                    name = metric
                    tf.summary.scalar(metric, metrics[metric], self.current_step())
        return


    def get_gradients(self, loss, tape, trainable_weights):

        return tape.gradient(loss, self._net.trainable_weights)

    @tf.function
    def apply_gradients(self, gradients):
        self._opt.apply_gradients(zip(gradients, self._net.trainable_variables))


    #
    # def create_model_summaries(self):
    #     # # return
    #     # optimizer = tf.train.AdamOptimizer(..)
    #     # grads = optimizer.compute_gradients(loss)
    #     # weights = self._net.trainable_variables
    #     # weight_sum_op = _accum_gradients
    #     # with tf.variable_scope)
    #     hist = []
    #
    #     self._accum_vars,
    #     for var, grad in zip(tf.compat.v1.trainable_variables(), self._accum_gradients):
    #         name = var.name.replace("/",".")
    #         hist.append(tf.summary.histogram(name, var))
    #         hist.append(tf.summary.histogram(name  + "/grad/", grad))
    #     # grad_summ_op = tf.summary.merge([tf.summary.histogram("%s-grad" % g[1].name, g[0]) for g in grads])
    #     # grad_vals = sess.run(fetches=grad_summ_op, feed_dict = feed_dict)
    #     self.model_summary = tf.summary.merge(hist)
    #     # self.model_summary = tf.compat.v1.summary.merge(hist)


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

        if self._val_writer is None:
            return

        gs = self.current_step()

        if gs % self.args.aux_iteration == 0:

            # Fetch the next batch of data with larcv
            minibatch_data = self.larcv_fetcher.fetch_next_batch('aux', force_pop = True)


            labels, logits, prediction = self.forward_pass(minibatch_data, training=False)

            loss = self.loss_calculator(labels, logits)



            metrics = self._compute_metrics(logits, prediction, labels, loss)


            # Report metrics on the terminal:
            self.log(metrics, kind="Test", step=int(self.current_step().numpy()))


            self.summary(metrics)
            self.summary_images(labels, prediction)

        return



    def train_step(self):

        global_start_time = datetime.datetime.now()

        io_fetch_time = 0.0

        gradients = None
        metrics = {}

        for i in range(self.args.gradient_accumulation):

            # Fetch the next batch of data with larcv
            io_start_time = datetime.datetime.now()
            minibatch_data = self.larcv_fetcher.fetch_next_batch("train",force_pop=True)
            io_end_time = datetime.datetime.now()
            io_fetch_time += (io_end_time - io_start_time).total_seconds()

            with self.tape:
                labels, logits, prediction = self.forward_pass(minibatch_data, training=True)

                loss = self.loss_calculator(labels, logits)

            # Do the backwards pass for gradients:
            if gradients is None:
                gradients = self.get_gradients(loss, self.tape, self._net.trainable_weights)
            else:
                gradients += self.get_gradients(loss, self.tape, self._net.trainable_weights)



            # Compute any necessary metrics:
            interior_metrics = self._compute_metrics(logits, prediction, labels, loss)

            for key in interior_metrics:
                if key in metrics:
                    metrics[key] += interior_metrics[key]
                else:
                    metrics[key] = interior_metrics[key]

        # Normalize the metrics:
        for key in metrics:
            metrics[key] /= self.args.gradient_accumulation

        # Add the global step / second to the tensorboard log:
        try:
            metrics['global_step_per_sec'] = 1./self._seconds_per_global_step
            metrics['images_per_second'] = self.args.minibatch_size / self._seconds_per_global_step
        except:
            metrics['global_step_per_sec'] = 0.0
            metrics['images_per_second'] = 0.0

        metrics['io_fetch_time'] = io_fetch_time

        # After the accumulation, weight the gradients as needed and apply them:
        if self.args.gradient_accumulation != 1:
            gradients /= self.args.gradient_accumulation

        self.apply_gradients(gradients)


        # Add the global step / second to the tensorboard log:
        try:
            metrics['global_step_per_sec'] = 1./self._seconds_per_global_step
            metrics['images_per_second'] = (self.args.minibatch_size*self.args.gradient_accumulation) / self._seconds_per_global_step
        except AttributeError:
            metrics['global_step_per_sec'] = 0.0
            metrics['images_per_second'] = 0.0



        self.summary(metrics)
        self.summary_images(labels, prediction)

        # Report metrics on the terminal:
        self.log(metrics, kind="Train", step=int(self.current_step().numpy()))


        global_end_time = datetime.datetime.now()

        # Compute global step per second:
        self._seconds_per_global_step = (global_end_time - global_start_time).total_seconds()

        # Update the global step:
        self._global_step.assign_add(1)
        # Update the learning rate:
        self._learning_rate.assign(self.lr_calculator(int(self._global_step.numpy())))
        return self.current_step()


    def current_step(self):
        return self._global_step


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
        ops['global_step'] = int(self._global_step.numpy())
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
        self.log(ops["metrics"], kind="Inference", step=self.current_step())

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
            fd.update({self._learning_rate : self.lr_calculator(self.current_step())})
        return fd

    def close_savers(self):
        pass
        # if self.args.mode == 'inference':
        #     if self.larcv_fetcher._writer is not None:
        #         self.larcv_fetcher._writer.finalize()
