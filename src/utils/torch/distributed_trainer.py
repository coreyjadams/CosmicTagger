import os
import sys
import time
from collections import OrderedDict

import numpy

import torch
import horovod.torch as hvd
hvd.init()

from ..core import flags
# from . import data_transforms
FLAGS = flags.FLAGS()


if not FLAGS.SYNTHETIC:
    from larcv.distributed_queue_interface import queue_interface

from mpi4py import MPI
comm = MPI.COMM_WORLD



from .trainer import torch_trainer

import tensorboardX


def lambda_warmup(epoch):
    # Constant terms:
    flat_warmup = 50
    linear_warmup = 50
    full = 10000
    size=hvd.size()
    target = numpy.sqrt(size)
    # Perform 500 warmup steps, gradually ramping the rate:
    if epoch <= flat_warmup:
        return 1.0
    elif epoch < flat_warmup + linear_warmup:
        return 1.0 + (target - 1) * (epoch - flat_warmup) / linear_warmup
    elif epoch <= flat_warmup + linear_warmup + full:
        return target
    else:
        return target * numpy.exp(-0.0001*(epoch-(full+linear_warmup+flat_warmup)))

def lr_increase(step):

    # This function actually ignores the input and uses the global step variable
    # This allows it to get the learning rate correct after restore.

    # For this problem, the dataset size is 1e5.
    # So the epoch can be calculated easily:
    # epoch = (step * FLAGS.MINIBATCH_SIZE) / (1e5)

    base_lr   = FLAGS.LEARNING_RATE
    step_size = 0.050

    return 1.0 + step*step_size

peak_lr = 0.5
cycle_len = 0.8

def one_cycle_clr(step):

    peak = peak_lr / FLAGS.LEARNING_RATE

    cycle_steps  = int(FLAGS.ITERATIONS*cycle_len)
    end_steps = FLAGS.ITERATIONS - cycle_steps
    # Which cycle are we in?

    cycle = int(step / cycle_steps)
    intra_step = 1.0 * (step % cycle_steps)

    base_multiplier = 1.0

    if cycle < 1:
#         base_multiplier *= 0.5

        if intra_step > cycle_steps*0.5:
            intra_step = cycle_steps - intra_step

        value = intra_step * (peak) /(0.5*cycle_steps)

    else:
        value = (intra_step / end_steps)*-1.0

    # print("Step: {}, Cycle: {}, base {}, intra_step {}, value: {}, total_scale: {}".format(
    #     step, cycle, base_multiplier, intra_step, value, base_multiplier + value)
    # )

    return base_multiplier + value


class distributed_trainer(torch_trainer):
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

        if FLAGS.COMPUTE_MODE == "GPU":
            os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())


        self._larcv_interface = queue_interface()
        self._iteration       = 0
        self._rank            = hvd.rank()
        self._cleanup         = []


        # Make sure that 'LEARNING_RATE' and 'TRAINING'
        # are in net network parameters:



    def save_model(self):

        if hvd.rank() == 0:
            torch_trainer.save_model(self)




    def init_optimizer(self):

        # This takes the base optimizer (self._opt) and replaces
        # it with a distributed version

        torch_trainer.init_optimizer(self)

        # Wrap the optimizer in a learning rate controller to ensure warmup and
        # decayed rate at the end.

        # Important: this lambda takes 'epoch' as an argument but it's actually
        # operating on the 'step' parameter.


        self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self._opt, one_cycle_clr, last_epoch=-1)


        self._opt = hvd.DistributedOptimizer(self._opt, named_parameters=self._net.named_parameters())




    def init_saver(self):
        if hvd.rank() == 0:
            torch_trainer.init_saver(self)
        else:
            self._saver = None
            self._aux_saver = None


    def initialize(self, io_only = False):

        print("HVD rank: {}".format(hvd.rank()))

        if self._rank == 0:
            FLAGS.dump_config()


        self._initialize_io(color=self._rank)


        if io_only:
            return

        self.init_network()


        if FLAGS.TRAINING:
            self._net.train(True)



        if hvd.rank() == 0:
            n_trainable_parameters = 0
            for var in self._net.parameters():
                n_trainable_parameters += numpy.prod(var.shape)
            print("Total number of trainable parameters in this network: {}".format(n_trainable_parameters))

        self.init_optimizer()

        self.init_saver()
        # hooks = self.get_standard_hooks()


        # Here, either restore the weights of the network or initialize it:
        self._global_step = 0
        # Restore the state from the root rank:
        if hvd.rank() == 0:
            state = self.restore_model()
        else:
            state = None

        if state is not None and hvd.rank() == 0:
            self.load_state(state)

        if FLAGS.MODEL_HALF_PRECISION:
            self._net.half()


        # Broadcast the state of the model:
        hvd.broadcast_parameters(self._net.state_dict(), root_rank = 0)


        if FLAGS.COMPUTE_MODE == "CPU":
            pass
        if FLAGS.COMPUTE_MODE == "GPU":
            self._net.cuda()

        # comm.bcast(self._global_step, root = 0)
        # hvd.broadcast_optimizer_state(self._opt, root_rank = 0)



    def summary(self, metrics, saver=""):
        if hvd.rank() == 0:
            torch_trainer.summary(self, metrics, saver)
        return

    def summary_images(self, logits_image, labels_image, saver=""):
        if hvd.rank() == 0:
            torch_trainer.summary_images(self, logits_image, labels_image, saver)
        return

    def _compute_metrics(self, logits, minibatch_data, loss):
        # This function calls the parent function which computes local metrics.
        # Then, it performs an all reduce on all metrics:
        metrics = torch_trainer._compute_metrics(self, logits, minibatch_data, loss)


        for key in metrics:
            # print("All reducing ", key)
            metrics[key] = hvd.allreduce(metrics[key], name = key)

        return metrics

    def on_step_end(self):
        self._lr_scheduler.step()
        pass


    # def to_torch(self, minibatch_data):

    #     # This function wraps the to-torch function but for a gpu forces
    #     # the right device
    #     if FLAGS.COMPUTE_MODE == 'GPU':
    #         device = torch.device('cuda')
    #         # device = torch.device('cuda:{}'.format(hvd.local_rank()))
    #     else:
    #         device = None
    #     minibatch_data = torch_trainer.to_torch(self, minibatch_data, device)

    #     return minibatch_data

    def log(self, metrics, saver=""):
        if hvd.rank() == 0:
            torch_trainer.log(self, metrics, saver)
