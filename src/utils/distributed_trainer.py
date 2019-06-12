import os
import sys
import time
from collections import OrderedDict

import numpy

import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
logging.getLogger("tensorflow").setLevel(logging.WARNING)
import tensorflow as tf

for i, p in enumerate(sys.path):
    if ".local" in p:
        sys.path.pop(i)

import horovod.tensorflow as hvd
hvd.init()

os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())

from larcv.distributed_larcv_interface import larcv_interface

from .trainercore import trainercore
from . import flags
FLAGS = flags.FLAGS()

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

        self._cleanup         = []


    def set_compute_parameters(self):

        self._config = tf.ConfigProto()


        if FLAGS.COMPUTE_MODE == "CPU":
            self._config.inter_op_parallelism_threads = FLAGS.INTER_OP_PARALLELISM_THREADS
            self._config.intra_op_parallelism_threads = FLAGS.INTRA_OP_PARALLELISM_THREADS
        if FLAGS.COMPUTE_MODE == "GPU":
            self._config.gpu_options.allow_growth = True
            # This is managed with the env variable at the top:
            # self._config.gpu_options.visible_device_list = str(hvd.local_rank())


    def initialize(self, io_only=False):



        tf.logging.info("HVD rank: {}".format(hvd.rank()))
        self.set_compute_parameters()

        self._initialize_io()

        if io_only:
            return


        if hvd.rank() == 0:
            print(FLAGS.dump_config())



        graph = tf.get_default_graph()
        self.init_network()

        if hvd.rank() == 0:
            self.print_network_info()



        self.init_optimizer()

        self.init_saver()

        # Take all of the metrics and turn them into summaries:
        for key in self._metrics:
            tf.summary.scalar(key, self._metrics[key])

        self._summary_basic = tf.summary.merge_all()
        self._summary_images = self._create_summary_images(self._input['label'], self._output['prediction'])



        # Add the graph to the log file:
        if hvd.rank() == 0:
            self._main_writer.add_graph(graph)


        # Create a session:
        self._sess = tf.Session(config = self._config)

        # Try to restore a model?
        restored = self.restore_model()

        if not restored and hvd.rank() == 0:
            self._sess.run(tf.global_variables_initializer())

        # Rank 0 has either restored, or has initialized.  Broadcast it:

        bcast = hvd.broadcast_global_variables(0)

        print(bcast)



        # NEED TO BROADCAST GLOBAL VARIABLES



        with tf.variable_scope("hvd"):

            # self._global_step = tf.train.get_or_create_global_step()


            # # All reduce metrics:
            # for key in self._metrics:
            #     self._metrics[key] = hvd.allreduce(self._metrics[key])

            # # Create a set of summary objects:
            # for key in self._metrics:
            #     tf.summary.scalar(key, self._metrics[key])

            # In the distributed case, we may want a learning rate behavior:
            self._lr = self.generate_learning_rate(FLAGS.LEARNING_RATE, self._global_step)



            # Wrap the optimizer it in horovod:
            self._opt = hvd.DistributedOptimizer(self._opt)

            self._train_op = self._opt.minimize(self._loss, self._global_step)

            # Here, we have to replace all of the metrics with the allreduced version:

            # Take all of the metrics and turn them into summaries:




    def restore_model(self):
        # Restore model has to restore on one rank and broadcast to other ranks
        if hvd.rank() == 0:
            restored = trainercore.restore_model(self)

        return restored



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


    def save_model(self, gs):
        if hvd.rank() != 0:
            return
        else:
            trainercore.save_model(self, gs)

    def write_summaries(self, writer, summary, global_step):
        if hvd.rank() != 0:
            return
        else
            trainercore.write_summaries(self, writer, summary, global_step)

    def log(self, metrics, kind, step):
        if hvd.rank() != 0:
            return
        else
            trainercore.log(self, metrics, kind, step)
   
    def batch_process(self):

        if hvd.rank() == 0:
            trainercore.batch_process(self, verbose=True)
        else:
            trainercore.batch_process(self, verbose=False)

