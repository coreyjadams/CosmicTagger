import os
import sys
import time
import tempfile
from collections import OrderedDict

from functools import reduce

import numpy


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

from src.utils.core.trainercore import trainercore
from src.networks.tensorflow    import uresnet2D, uresnet3D, LossCalculator, AccuracyCalculator


import datetime



import tensorflow as tf
tf.get_logger().setLevel('INFO')

tf.get_logger().setLevel('INFO')


floating_point_format = tf.float32
integer_format = tf.int64

from src.utils import logging
logger = logging.getLogger("CosmicTagger")

from src.config import Precision, ComputeMode, ModeKind, DataFormatKind



class tf_trainer(trainercore):
    '''
    This is the tensorflow version of the trainer

    '''

    def __init__(self, args, datasets, lr_schedule, log_keys, hparams_keys):
        trainercore.__init__(self, args)

        self.datasets = datasets
        self.lr_schedule = lr_schedule
        self.log_keys = log_keys
        self.hparams_keys  = hparams_keys

        self._channels_dim = -1
        if self.args.data.data_format == DataFormatKind.channels_first:
            self._channels_dim = 1

        # self.initialize(datasets)

    def local_batch_size(self):
        return self.args.run.minibatch_size

    def init_network(self):

        # This function builds the compute graph.
        # Optionally, it can build a 'subset' graph if this mode is

        # Net construction:
        start = time.time()

        # Here, if using mixed precision, set a global policy:
        if self.args.run.precision == Precision.mixed:
            from tensorflow.keras import mixed_precision
            self.policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(self.policy)

        if self.args.run.precision == Precision.bfloat16:
            from tensorflow.keras import mixed_precision
            self.policy = mixed_precision.Policy('mixed_bfloat16')
            mixed_precision.set_global_policy(self.policy)

        #
        self._global_step = tf.Variable(0, dtype=tf.int64)


        # Add the dataformat for the network construction:

        example_dataset = next(iter(self.datasets.values()))

        from src.config import ConvMode
        # Build the network object, forward pass only:
        if self.args.network.conv_mode == ConvMode.conv_2D:
            self._net = uresnet2D.UResNet(self.args.network, example_dataset.image_size)
        else:
            self._net = uresnet3D.UResNet3D(self.args.network, example_dataset.image_size)

        self._net.trainable = True


        # TO PROPERLY INITIALIZE THE NETWORK, NEED TO DO A FORWARD PASS
        minibatch_data = next(iter(example_dataset))
        image, label = self.cast_input(minibatch_data['image'], minibatch_data['label'])

        self.forward_pass(image, label)

        self.acc_calculator  = AccuracyCalculator.AccuracyCalculator()
        if self.is_training():
            reg_loss_fn = self._net.reg_loss
            self.loss_calculator = LossCalculator.LossCalculator(reg_loss_fn,
                self.args.mode.optimizer.loss_balance_scheme, self._channels_dim)


        self._log_keys = ["loss/loss", "Average/Non_Bkg_Accuracy", "Average/mIoU"]

        end = time.time()
        return end - start

    def print_network_info(self, verbose=False):
        n_trainable_parameters = 0

        # Are we trying to run in deterministic mode?  If so, print out
        # Extra info on the weights:
        deterministic = self.args.framework.seed != 0 and self.args.data.seed != 0


        for var in self._net.variables:
            n_trainable_parameters += numpy.prod(var.shape)
            if deterministic:
                logger.info(f"{var.name} has min/mean/max {tf.reduce_min(var)}/{tf.reduce_mean(var)}/{tf.reduce_max(var)}.")
                logger.info(f"{var.device}")
            if verbose:
                logger.info(f"{var.name}: {var.shape()}")
        logger.info(f"Total number of trainable parameters in this network: {n_trainable_parameters}")

    def current_step(self):
        return self._global_step


    def set_compute_parameters(self):


        if self.args.run.compute_mode == ComputeMode.CPU:
            self._config.inter_op_parallelism_threads = self.args.framework.inter_op_parallelism_threads
            self._config.intra_op_parallelism_threads = self.args.framework.intra_op_parallelism_threads
        elif self.args.run.compute_mode == ComputeMode.CUDA:
            gpus = tf.config.experimental.list_physical_devices('CUDA')
        elif self.args.run.compute_mode == ComputeMode.XPU:
            gpus = tf.config.experimental.list_physical_devices('XPU')


            # The code below is for MPS mode.  It is a bit of a hard-coded
            # hack.  Use with caution since the memory limit is set by hand.
            ####################################################################
            # print(gpus)
            # if gpus:
            #   # Restrict TensorFlow to only allocate 1GB of memory on the first CUDA
            #   try:
            #     tf.config.experimental.set_virtual_device_configuration(
            #         gpus[0],
            #         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15000)])
            #     # tf.config.experimental.set_memory_growth(gpus[0], True)
            #     logical_gpus = tf.config.experimental.list_logical_devices('CUDA')
            #     # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            #   except RuntimeError as e:
            #     # Virtual devices must be set before GPUs have been initialized
            #     print(e)
            ####################################################################

    # @tf.function
    def summary_images(self, labels, prediction, saver=""):
        ''' Create images of the labels and prediction to show training progress
        '''
        if self.current_step() % 25 * self.args.mode.summary_iteration == 0 and not self.args.mode.no_summary_images:


            if saver == "":
                saver = self.savers["train"]
            with saver.as_default():
                for p in range(len(labels)):
                    shape = labels[p][0].shape
                    label = tf.reshape( labels[p][0] / 2, (1,) + shape + (1,))
                    # Save only one event per snapshot
                    tf.summary.image(f"label/plane_{p}", label,     self.current_step())
                    pred = tf.reshape(prediction[p][0] / 2, (1,) + shape  + (1,))
                    tf.summary.image(f"pred/plane_{p}",  pred, self.current_step())


        return

    def initialize(self, datasets):

        self.set_compute_parameters()


        start = time.time()

        net_time = self.init_network()

        logger.info("Done constructing network. ({0:.2}s)\n".format(time.time()-start))


        self.print_network_info()

        if self.args.mode.name == ModeKind.train:
            self.init_optimizer()

        # self.init_saver()
        # Initialize savers:
        dir = self.args.output_dir
        if not self.args.run.distributed or self.rank == 0:
            self.savers = {
                ds_name : tf.summary.create_file_writer(dir + f"/{ds_name}/")
                for ds_name in datasets.keys()
            }
        else:
            self.savers = {ds_name : None for ds_name in datasets.keys()}


        # Try to restore a model?
        restored = self.restore_model()

        if self.args.mode.name == ModeKind.inference:
            self.inference_metrics = {}
            self.inference_metrics['n'] = 0


    def init_learning_rate(self):
        # Use a place holder for the learning rate :
        # self._learning_rate = tf.Variable(
        #     initial_value=0.0, 
        #     trainable=False, 
        #     dtype=floating_point_format, 
        #     name="lr"
        # )
        self._learning_rate = self.lr_schedule

    def restore_model(self):
        ''' This function attempts to restore the model from file
        '''


        def check_inference_weights_path(file_path):
            # Look for the "checkpoint" file:
            checkpoint_file_path = file_path + "checkpoint"
            # If it exists, open it and read the latest checkpoint:
            if os.path.isfile(checkpoint_file_path):
                return file_path
            else:
                return None


        # First, check if the weights path is set:
        if self.args.mode.name == ModeKind.inference and self.args.mode.weights_location != "":
            file_path = check_inference_weights_path(self.args.mode.weights_location)
        else:
            file_path = self.get_checkpoint_dir()



        path = tf.train.latest_checkpoint(file_path)

        if path is None:
            logger.info("No checkpoint found, starting from scratch")
            return False
        # Parse the checkpoint file and use that to get the latest file path
        logger.info(f"Restoring checkpoint from {path}")
        self._net.load_weights(path)
        # self.scheduler.set_current_step(self.current_step())

        return True

    def checkpoint(self):

        gs = int(self.current_step().numpy())

        if gs % self.args.mode.checkpoint_iteration == 0 and gs != 0:
            # Save a checkpoint, but don't do it on the first pass
            self.save_model(gs)

    def get_checkpoint_dir(self):

        # Find the base path of the log directory
        file_path= self.args.output_dir + "/checkpoints/"

        return file_path

    def save_model(self, global_step):
        '''Save the model to file

        '''

        file_path = self.get_checkpoint_dir()

        # Make sure the path actually exists:
        if not os.path.isdir(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        saved_path = self._net.save_weights(file_path + "model_{}.weights.h5".format(global_step))


    def get_model_filepath(self, global_step):
        '''Helper function to build the filepath of a model for saving and restoring:

        '''

        file_path = self.get_checkpoint_dir()

        name = file_path + 'model-{}.weights.h5'.format(global_step)
        checkpoint_file_path = file_path + "checkpoint"

        return name, checkpoint_file_path


    def init_optimizer(self):

        if self.args.mode.name != ModeKind.train: return

        from src.config import OptimizerKind

        self.init_learning_rate()


        if self.args.mode.optimizer.name == OptimizerKind.rmsprop:
            # Use RMS prop:
            self._opt = tf.keras.optimizers.RMSprop(self._learning_rate)
        else:
            # default is Adam:
            self._opt = tf.keras.optimizers.Adam(self._learning_rate)

        if self.args.run.precision == Precision.mixed:
            self._opt = tf.keras.mixed_precision.LossScaleOptimizer(self._opt)


        self.tape = tf.GradientTape()

    def _compute_metrics(self, logits, prediction, labels, loss, reg_loss):

        # self._output['softmax'] = [ tf.nn.softmax(x) for x in self._logits]
        # self._output['prediction'] = [ tf.argmax(input=x, axis=self._channels_dim) for x in self._logits]
        # If the mode is channels last,
        accuracy = self.acc_calculator(prediction=prediction, labels=labels)


        metrics = {}
        for p in [0,1,2]:
            metrics[f"plane{p}/Total_Accuracy"]          = accuracy["total_accuracy"][p]
            metrics[f"plane{p}/Non_Bkg_Accuracy"]        = accuracy["non_bkg_accuracy"][p]
            metrics[f"plane{p}/Neutrino_IoU"]            = accuracy["neut_iou"][p]
            metrics[f"plane{p}/Cosmic_IoU"]              = accuracy["cosmic_iou"][p]

        metrics["Average/Total_Accuracy"]          = tf.reduce_mean(accuracy["total_accuracy"])
        metrics["Average/Non_Bkg_Accuracy"]        = tf.reduce_mean(accuracy["non_bkg_accuracy"])
        metrics["Average/Neutrino_IoU"]            = tf.reduce_mean(accuracy["neut_iou"])
        metrics["Average/Cosmic_IoU"]              = tf.reduce_mean(accuracy["cosmic_iou"])
        metrics["Average/mIoU"]                    = tf.reduce_mean(accuracy["miou"])

        if loss is not None:
            metrics['loss/total'] = loss
        if reg_loss is not None:
            metrics['loss/reg_loss'] = reg_loss

        return metrics

    def log(self, metrics, keys, saver):
        metrics = { key : float(val) for key, val in metrics.items()}
        trainercore.log(self, metrics, keys, saver)

    # @tf.function
    def cast_input(self, image, label):
        if self.args.run.precision == Precision.float32 or self.args.run.precision == Precision.mixed:
            input_dtype = tf.float32
        elif self.args.run.precision == Precision.bfloat16:
            input_dtype = tf.bfloat16

        image = tf.convert_to_tensor(image, dtype=input_dtype)
        label = tf.convert_to_tensor(label, dtype=tf.int32)

        return image, label

    @tf.function
    def forward_pass(self, image, label):

        if self.args.run.precision == Precision.bfloat16:
            image = tf.cast(image, tf.bfloat16)

        # Run a forward pass of the model on the input image:
        logits = self._net(image)

        if self.args.run.precision == Precision.mixed:
            logits = [ tf.cast(l, tf.float32) for l in logits ]
        # elif self.args.run.precision == Precision.bfloat16:
        #     logits = [ tf.cast(l, tf.bfloat16) for l in logits ]

        prediction = [ tf.argmax(l, axis=self._channels_dim, output_type = tf.dtypes.int32) for l in logits ]
        labels = tf.split(label, num_or_size_splits=3, axis=self._channels_dim)
        labels = [tf.squeeze(li, axis=self._channels_dim) for li in labels]

        return labels, logits, prediction

    # @tf.function(experimental_relax_shapes=True)
    def summary(self, metrics, saver=""):

        if self.current_step() % self.args.mode.summary_iteration == 0:
            if saver == "":
                saver = self.savers['train']

            with saver.as_default():
                for metric in metrics:
                    name = metric
                    tf.summary.scalar(metric, metrics[metric], self.current_step())
        return

    # @tf.function
    def get_gradients(self, loss, tape, trainable_weights):
        return tape.gradient(loss, self._net.trainable_weights)

    @tf.function
    def apply_gradients(self, gradients):
        self._opt.apply_gradients(zip(gradients, self._net.trainable_variables))




    def metrics(self, metrics):
        # This function looks useless, but it is not.
        # It allows a handle to the distributed network to allreduce metrics.
        return metrics

    def val_step(self, minibatch_data, store=True):


        if self.args.data.synthetic:
            return


        image, label = self.cast_input(minibatch_data['image'], minibatch_data['label'])

        labels, logits, prediction = self.forward_pass(image, label)

        loss, current_reg_loss = self.loss_calculator(labels, logits)

        metrics = self._compute_metrics(logits, prediction, labels, loss, current_reg_loss)


        if store:
            # Report metrics on the terminal:
            self.log(metrics, self.log_keys, saver="val")


            self.summary(metrics=metrics, saver=self.savers['val'])
            self.summary_images(labels, prediction, saver=self.savers['val'])

        return


    @tf.function
    def gradient_step(self, image, label):

        with self.tape:
            with self.timing_context("forward"):
                labels, logits, prediction = self.forward_pass(image, label)

            with self.timing_context("loss"):

                # The loss function has to be in full precision or automatic mixed.
                # bfloat16 is not supported
                if self.args.run.precision == Precision.bfloat16:
                    logits = [ tf.cast(l, dtype=tf.float32) for  l in logits ]

                loss, reg_loss = self.loss_calculator(labels, logits)
            #
                # loss = loss + reg_loss

                if self.args.run.precision == Precision.mixed:
                    scaled_loss = self._opt.scale_loss(loss)

        # Do the backwards pass for gradients:
        with self.timing_context("backward"):
            if self.args.run.precision == Precision.mixed:
                scaled_gradients = self.get_gradients(scaled_loss, self.tape, self._net.trainable_weights)
                gradients = self._opt.get_unscaled_gradients(scaled_gradients)
            else:
                gradients = self.get_gradients(loss, self.tape, self._net.trainable_weights)
        # TODO: Should this add reg_loss to the loss?  Need to check how TF handles this.
        return logits, labels, prediction, loss, gradients, reg_loss
        # return logits, labels, prediction, loss - reg_loss, gradients, reg_loss

    def train_step(self, minibatch_data):


        global_start_time = datetime.datetime.now()

        io_fetch_time = 0.0

        gradients = None
        metrics = {}

        gradient_accumulation = self.args.mode.optimizer.gradient_accumulation
        for i in range(gradient_accumulation):

            # Fetch the next batch of data with larcv
            io_start_time = datetime.datetime.now()

            image, label = self.cast_input(minibatch_data['image'], minibatch_data['label'])

            io_end_time = datetime.datetime.now()
            io_fetch_time += (io_end_time - io_start_time).total_seconds()

            if self.args.run.profile:
                if not self.args.run.distributed or self._rank == 0:
                    tf.profiler.experimental.start(self.args.output_dir + "/train/")
            logits, labels, prediction, loss, internal_gradients, reg_loss = self.gradient_step(image, label)

            if self.args.run.profile:
                if not self.args.run.distributed or self._rank == 0:
                    tf.profiler.experimental.stop()

            # Accumulate gradients if necessary
            if gradients is None:
                gradients = internal_gradients
            else:
                gradients += internal_gradients

            # Compute any necessary metrics:
            with self.timing_context("metrics"):
                interior_metrics = self._compute_metrics(logits, prediction, labels, loss, reg_loss)

                for key in interior_metrics:
                    if key in metrics:
                        metrics[key] += interior_metrics[key]
                    else:
                        metrics[key] = interior_metrics[key]



        # Normalize the metrics:
        for key in metrics:
            metrics[key] /= gradient_accumulation

        # Add the global step / second to the tensorboard log:
        try:
            metrics['global_step_per_sec'] = 1./self._seconds_per_global_step
            metrics['images_per_second'] = self.args.run.minibatch_size / self._seconds_per_global_step
        except:
            metrics['global_step_per_sec'] = 0.0
            metrics['images_per_second'] = 0.0

        metrics['io_fetch_time'] = io_fetch_time
        metrics['learning_rate'] = self._learning_rate(self._global_step)

        # After the accumulation, weight the gradients as needed and apply them:
        if gradient_accumulation != 1:
            gradients = [ g / gradient_accumulation for g in gradients ]

        with self.timing_context("optimizer"):
            self.apply_gradients(gradients)
        # print(self._net.trainable_variables)


        # Add the global step / second to the tensorboard log:
        try:
            metrics['global_step_per_sec'] = 1./self._seconds_per_global_step
            metrics['images_per_second'] = (self.args.run.minibatch_size*gradient_accumulation) / self._seconds_per_global_step
        except AttributeError:
            metrics['global_step_per_sec'] = 0.0
            metrics['images_per_second'] = 0.0


        with self.timing_context("summary"):
            self.summary(metrics)
            self.summary_images(labels, prediction)

        # with self.timing_context("log"):
        #     # Report metrics on the terminal:
        #     # self.log(metrics, kind="Train", step=int(self.current_step().numpy()))
        #     self.log(metrics, self.log_keys, saver="train")


        global_end_time = datetime.datetime.now()

        # Compute global step per second:
        self._seconds_per_global_step = (global_end_time - global_start_time).total_seconds()

        # Update the global step:
        self._global_step.assign_add(1)
        # Update the learning rate:
        # self._learning_rate.assign(self.lr_schedule(int(self._global_step.numpy())))
        # self._opt.lr.assign(self._learning_rate)

        return metrics


    def current_step(self):
        return self._global_step


    def stop(self):
        # Mostly, this is just turning off the io:
        # self._larcv_interface.stop()
        pass


    def ana_step(self, minibatch_data):


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

        image, label = self.cast_input(minibatch_data['image'], minibatch_data['label'])

        io_end_time   = datetime.datetime.now()
        io_fetch_time = (io_end_time - io_start_time).total_seconds()

        if self.args.run.profile:
            if not self.args.distributed or self._rank == 0:
                tf.profiler.experimental.start(self.args.output_dir + "/train/")
        # Get the logits:
        labels, logits, prediction = self.forward_pass(image, label)

        # # And the loss:
        # loss = self.loss_calculator(labels, logits)


        if self.args.run.profile:
            if not self.args.distributed or self._rank == 0:
                tf.profiler.experimental.stop()



        # Compute any necessary metrics:
        metrics = self._compute_metrics(logits, prediction, labels, loss=None, reg_loss=None)


        if tf.math.is_nan(metrics['Average/mIoU']):
            for key in metrics:
                print(f"{key}: {metrics[key]}")

        self.accumulate_metrics(metrics)


        # Add the global step / second to the tensorboard log:
        try:
            metrics['global_step_per_sec'] = 1./self._seconds_per_global_step
            metrics['images_per_second'] = self.args.run.minibatch_size / self._seconds_per_global_step
        except AttributeError:

            metrics['global_step_per_sec'] = 0.0
            metrics['images_per_second'] = 0.0

        metrics['io_fetch_time'] = io_fetch_time


        # Report metrics on the terminal:
        # self.log(metrics, kind="Inference", step=int(self.current_step().numpy()))
        self.log(metrics, self.log_keys, saver="Inference")


        self.summary(metrics)
        self.summary_images(labels, prediction)
        self._global_step.assign_add(1)



        global_end_time = datetime.datetime.now()

        # Compute global step per second:
        self._seconds_per_global_step = (global_end_time - global_start_time).total_seconds()
        # self._global_step += 1

        return

    def accumulate_metrics(self, metrics):

        self.inference_metrics['n'] += 1
        for key in metrics:
            if key not in self.inference_metrics:
                self.inference_metrics[key] = metrics[key]
                # self.inference_metrics[f"{key}_sq"] = metrics[key]**2
            else:
                self.inference_metrics[key] += metrics[key]
                # self.inference_metrics[f"{key}_sq"] += metrics[key]**2

    def inference_report(self):
        if not hasattr(self, "inference_metrics"):
            return
        n = self.inference_metrics["n"]
        total_entries = n*self.args.run.minibatch_size
        logger.info(f"Inference report: {n} batches processed for {total_entries} entries.")
        for key in self.inference_metrics:
            if key == 'n' or '_sq' in key: continue
            value = self.inference_metrics[key] / n
            logger.info(f"  {key}: {value:.4f}")


    def close_savers(self):
        pass
