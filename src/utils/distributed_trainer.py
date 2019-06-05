import os
import sys
import time
from collections import OrderedDict

import numpy

import tensorflow as tf

for i, p in enumerate(sys.path):
    if ".local" in p:
        sys.path.pop(i)

import horovod.tensorflow as hvd
hvd.init()

from larcv.distributed_larcv_interface import larcv_interface

from .uresnet import uresnet
from .trainercore import trainercore
from .flags import FLAGS

class distributed_trainer(trainercore):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self):
        # Rely on the base class for most standard parameters, only
        # search for parameters relevant for distributed computing here

        # Put the IO rank as the last rank in the COMM, since rank 0 does tf saves
        root_rank = hvd.size() - 1 

        self._larcv_interface = larcv_interface(root=root_rank)
        self._iteration       = 0
        self._rank            = hvd.rank()

        self._val_writer      = None

        # Make sure that 'LEARNING_RATE' and 'TRAINING'
        # are in net network parameters:

        self._metrics = OrderedDict()


        self._initialize()



    def initialize(self, io_only=False):


        tf.logging.info("HVD rank: {}".format(hvd.rank()))

        self._initialize_io()



        if io_only:
            return

        if hvd.rank() == 0:
            tf.logging.info(FLAGS.get_config())


        graph = tf.get_default_graph()
        self.init_network()

        self.print_network_info()

        self.init_optimizer()


        self.set_compute_parameters()


        with tf.variable_scope("hvd"):

            # self._global_step = tf.train.get_or_create_global_step()


            # # All reduce metrics:
            for key in self._metrics:
                self._metrics[key] = hvd.allreduce(self._metrics[key])

            # Create a set of summary objects:
            for key in self._metrics:
                tf.summary.scalar(key, self._metrics[key])

            # In the distributed case, we may want a learning rate behavior:
            self._lr = self.generate_learning_rate(FLAGS.LEARNING_RATE, self._global_step)


            # Only compute summaries on the root node:
            if hvd.rank() == 0:
                # Merge the summary
                self._summary_basic = tf.summary.merge_all()

                # Store the predicted pixels into a tf summary:
                self._summary_images = self._create_summary_images(self._input['label'], self._outputs['prediction'])


            # Wrap the optimizer it in horovod:
            self._opt = hvd.DistributedOptimizer(self._opt)

            self._train_op = self._opt.minimize(self._loss, self._global_step)

            # Here, we have to replace all of the metrics with the allreduced version:

            # Take all of the metrics and turn them into summaries:

            # Additionally, in training mode if there is aux data use it for validation:
            if hvd.rank() == 0 and FLAGS.AUX_FILE is not None:
                self._val_writer = tf.summary.FileWriter(logdir=FLAGS.LOG_DIRECTORY+"/test/")

        hooks = self.get_distributed_hooks()




        if hvd.rank() == 0:
            if FLAGS.CHECKPOINT_ITERATION > 0:
                checkpoint_it = FLAGS.CHECKPOINT_ITERATION
            else:
                checkpoint_it = None
            self._sess = tf.train.MonitoredTrainingSession(config=config, hooks = hooks,
                checkpoint_dir        = FLAGS.LOG_DIRECTORY,
                log_step_count_steps  = FLAGS.LOGGING_ITERATION,
                save_summaries_steps  = None,
                save_summaries_secs   = None,
                save_checkpoint_secs  = None,
                save_checkpoint_steps = checkpoint_it
            )

        else:
            self._sess = tf.train.MonitoredTrainingSession(config=config, hooks = hooks,
                checkpoint_dir = None,
                save_summaries_steps = None,
                save_summaries_secs = None,
            )

    def get_distributed_hooks(self):

        if hvd.rank() == 0:

            checkpoint_dir = FLAGS.LOG_DIRECTORY

            hooks = [hvd.BroadcastGlobalVariablesHook(0)]

            # reduce_metrics_hook = ReduceMetricsHook(
            #     metrics=self._metrics
            # )
            # hooks.append(reduce_metrics_hook)


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
                                  'accuracy'    : self._metrics['accuracy/All_Plane_Neutrino_Accuracy'], 
                                  'loss'        : self._metrics['cross_entropy/Total_Loss']},
                every_n_iter  = FLAGS.LOGGING_ITERATION,
                )
            hooks.append(logging_hook)



        else:            
            hooks = [
                hvd.BroadcastGlobalVariablesHook(0),
            ]

            # reduce_metrics_hook = ReduceMetricsHook(
            #     metrics=self._metrics
            # )
            # hooks.append(reduce_metrics_hook)

        return hooks


    def generate_learning_rate(self, 
        base_learning_rate, 
        global_step,
        warmup_steps = 100,
        decay_after_step=4500):

        ''' Compute the peak learning rate, the start point, and such
        '''

        # Learning rate scales linearly from a very low value to the base learning rate *sqrt(N)
        # for the duration of the warm up steps.

        # After some number of steps, the learning rate undergoes a decay by 10

        # For the calculations, we need to set some constants:

        core_learning_rate = tf.constant(base_learning_rate*numpy.sqrt(hvd.size()), dtype=tf.float32)
        initial_learning_rate = tf.constant(0.0001, dtype=tf.float32)
        warmup_steps = tf.constant(warmup_steps, dtype = tf.float32)


        # So, there are 3 phases: warm up, steady state, decay


        # First, have to decide what state we are in.

        scaled_learning_rate = initial_learning_rate +  core_learning_rate * tf.cast(global_step, tf.float32) / warmup_steps

        # Warm up phase:
        this_learning_rate = tf.math.minimum(scaled_learning_rate, core_learning_rate)


        # # Cool down phase:
        # this_learning_rate = tf.cond( global_step > decay_after_step,
        #     lambda: 0.1* core_learning_rate,
        #     lambda: this_learning_rate
        #     )

        lr_summary = tf.summary.scalar("LearningRate", this_learning_rate)

        # # Need to add this directly to the merged summary op:
        # self._summary_basic = tf.summary.merge([lr_summary, self._summary_basic])

        return this_learning_rate


