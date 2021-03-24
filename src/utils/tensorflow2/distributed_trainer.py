import os
import sys
import time
from collections import OrderedDict

import numpy

from .trainer import tf_trainer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

for i, p in enumerate(sys.path):
    if ".local" in p:
        sys.path.pop(i)

import horovod.tensorflow as hvd
hvd.init()

# from horovod.tensorflow.keras import DistributedOptimizer




class distributed_trainer(tf_trainer):
    '''a
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self, args):

        # Rely on the base class for most standard parameters, only
        # search for parameters relevant for distributed computing here
        tf_trainer.__init__(self, args)

        if self.args.compute_mode == "GPU":
            gpus = tf.config.list_physical_devices('GPU')
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

            # os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())

        self._rank            = hvd.rank()
        self.local_minibatch_size = int(self.args.minibatch_size / hvd.size())

    def print(self, *argv):

        if self._rank == 0:
            tf_trainer.print(self, *argv)

    def init_optimizer(self):
                # with tf.variable_scope("hvd"):

        #     # In the distributed case, we may want a learning rate behavior:
        #     self._learning_rate = self.generate_learning_rate(self.args.learning_rate, self._global_step)
        tf_trainer.init_optimizer(self)

        # Wrap the optimizer it in horovod:
        # self._opt = hvd.DistributedOptimizer(self._opt)
        self.tape = hvd.DistributedGradientTape(self.tape)

    def init_saver(self):
        if hvd.rank() == 0:
            tf_trainer.init_saver(self)
        else:
            self._saver = None
            self._aux_saver = None
            self._main_writer = None
            self._val_writer = None


    # def get_gradients(self, loss, tape, trainable_variables):

    #     print(gradients[0])
    #     tape = hvd.DistributedGradientTape(tape)
    #     gradients =  tape.gradient(loss, self._net.trainable_variables)

    #     print(gradients[0])

    #     return gradients

    # def apply_gradients(self, gradients):
    #     print(gradients)
    #     gradients = [ hvd.allreduce(gradient) for gradient in gradients ]

    #     tf_trainer.apply_gradients(self, gradients)

    def init_checkpointer(self):
        if hvd.rank() == 0:
            tf_trainer.init_checkpointer(self)
        else:
            self._checkpoint = None
            self._checkpoint_manager = None

    def initialize(self, io_only=False):


        tf_trainer.initialize(self, io_only)

        # Here, we broadcast parameters from rank 0.

        # If the model was restored, this is correct.  If not,
        # This syncs everythign up.
        # print(bcast)

        hvd.broadcast_variables(self._net.variables, root_rank=0)
        hvd.broadcast_variables(self._opt.variables(), root_rank=0)

    def restore_model(self):

        if self._rank == 0:
            # Restore the model on the root node:
            tf_trainer.restore_model(self)

    def write_graph_to_tensorboard(self, graph):
        if self._rank == 0:
            # Add the graph to the log file:
            tf_trainer.write_graph_to_tensorboard(self, graph)


    def local_batch_size(self):
        # If synthetic, the local batch size is the minibatchsize.

        # Otherwise, it's minibatch size / n_ranks:

        if self.args.synthetic:
            return self.args.minibatch_size
        else:
            lbs = int(self.args.minibatch_size / hvd.size())
            return lbs


    def restore_model(self):
        # Restore model has to restore on one rank and broadcast to other ranks
        if hvd.rank() == 0:
            restored = tf_trainer.restore_model(self)
            return restored
        else:
            return None


    def save_model(self, gs):
        if hvd.rank() != 0:
            return
        else:
            tf_trainer.save_model(self, gs)

    def summary(self, metrics, saver=""):
        if hvd.rank() != 0:
            return
        else:
            tf_trainer.summary(self, metrics, saver)

    def summary_images(self, labels, prediction, saver=""):
        if hvd.rank() != 0:
            return
        else:
            tf_trainer.summary_images(self, labels, prediction, saver)

    def log(self, metrics, kind, step):
        if hvd.rank() != 0:
            return
        else:
            tf_trainer.log(self, metrics, kind, step)

    # def batch_process(self):

    #     if hvd.rank() == 0:
    #         tf_trainer.batch_process(self)
    #     else:
    #         tf_trainer.batch_process(self)
