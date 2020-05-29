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

        sys.stdout.write(f"Rank {self._rank} entering the __init__ barrier")
        sys.stdout.flush()
        comm.Barrier()


        # Make sure that 'LEARNING_RATE' and 'TRAINING'
        # are in net network parameters:



    def save_model(self):

        if hvd.rank() == 0:
            torch_trainer.save_model(self)




    def init_optimizer(self):

        # This takes the base optimizer (self._opt) and replaces
        # it with a distributed version


        torch_trainer.init_optimizer(self)

        self._opt = hvd.DistributedOptimizer(self._opt, named_parameters=self._net.named_parameters())
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self._opt, self.lr_calculator, last_epoch=-1)


    def print(self, *argv):
        if self._rank == 0:
            torch_trainer.print(self, *argv)

            
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
            self.restore_state(state)



        # Broadcast the global step:
        self._global_step = hvd.broadcast_object(self._global_step, root_rank = 0)

        # Broadcast the state of the model:
        hvd.broadcast_parameters(self._net.state_dict(), root_rank = 0)

        # Broadcast the optimizer state:
        hvd.broadcast_optimizer_state(self._opt, root_rank = 0)

        # Broadcast the LR Schedule state:
        state_dict = hvd.broadcast_object(self.lr_scheduler.state_dict(), root_rank = 0)
        self.lr_scheduler.load_state_dict(state_dict)


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
            metrics[key] = hvd.allreduce(metrics[key], name = key)


        return metrics

    # def on_step_end(self):
    #     # self.lr_scheduler.step()
    #     pass



    def log(self, metrics, saver=""):

        if hvd.rank() == 0:
            torch_trainer.log(self, metrics, saver)
