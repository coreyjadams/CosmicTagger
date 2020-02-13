import os
import sys
import time
from collections import OrderedDict

import numpy

import torch
import horovod.torch as hvd
hvd.init()


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

# def lr_increase(step):

#     # This function actually ignores the input and uses the global step variable
#     # This allows it to get the learning rate correct after restore.

#     # For this problem, the dataset size is 1e5.
#     # So the epoch can be calculated easily:
#     # epoch = (step * self.args.MINIBATCH_SIZE) / (1e5)

#     base_lr   = self.args.LEARNING_RATE
#     step_size = 0.050

#     return 1.0 + step*step_size

# def flat_lr(step):
#     return 1.0

# peak_lr = 0.01
# cycle_len = 0.8

# def one_cycle_clr(step):

#     peak = peak_lr / self.args.LEARNING_RATE

#     cycle_steps  = int(self.args.ITERATIONS*cycle_len)
#     end_steps = self.args.ITERATIONS - cycle_steps
#     # Which cycle are we in?

#     cycle = int(step / cycle_steps)
#     intra_step = 1.0 * (step % cycle_steps)

#     base_multiplier = 1.0

#     if cycle < 1:
# #         base_multiplier *= 0.5

#         if intra_step > cycle_steps*0.5:
#             intra_step = cycle_steps - intra_step

#         value = intra_step * (peak) /(0.5*cycle_steps)

#     else:
#         value = (intra_step / end_steps)*-1.0

#     # print("Step: {}, Cycle: {}, base {}, intra_step {}, value: {}, total_scale: {}".format(
#     #     step, cycle, base_multiplier, intra_step, value, base_multiplier + value)
#     # )

#     return base_multiplier + value


class distributed_trainer(torch_trainer):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self, args):
        
        torch_trainer.__init__(self, args)
        # Rely on the base class for most standard parameters, only
        # search for parameters relevant for distributed computing here



        # Put the IO rank as the last rank in the COMM, since rank 0 does tf saves

        if self.args.compute_mode == "GPU":
            os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())

        self._rank            = hvd.rank()


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


        # self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            # self._opt, flat_lr, last_epoch=-1)


        self._opt = hvd.DistributedOptimizer(self._opt, named_parameters=self._net.named_parameters())




    def init_saver(self):
        if hvd.rank() == 0:
            torch_trainer.init_saver(self)
        else:
            self._saver = None
            self._aux_saver = None


    def print_network_info(self):
        if hvd.rank() == 0:
            torch_trainer.print_network_info(self)
        return

    def restore_model(self):
        if hvd.rank() == 0:
            state = self.load_state_from_file()
        else:
            state = None

        if state is not None and hvd.rank() == 0:
            self.load_state(state)

        # Broadcast the state of the model:
        hvd.broadcast_parameters(self._net.state_dict(), root_rank = 0)

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
        # self._lr_scheduler.step()
        pass



    def log(self, metrics, saver=""):
        if hvd.rank() == 0:
            torch_trainer.log(self, metrics, saver)
