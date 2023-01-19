import os
import sys
import time, datetime
from collections import OrderedDict

import contextlib



import numpy

import torch

# Always want mpi, but horovod is imported below.
from mpi4py import MPI
comm = MPI.COMM_WORLD

try:
    import torch_ipex as ipex
except:
    pass




# set IPEX XPU device before importing IPEX
try:
    import horovod.torch as hvd
    hvd.init()
    IPEX_TILE_AS_DEVICE = os.environ.get("IPEX_TILE_AS_DEVICE", "0")
    if IPEX_TILE_AS_DEVICE == "1":
        os.environ["IPEX_DEV_INDEX"] = str(hvd.local_rank())
    else:
        os.environ["ZE_AFFINITY_MASK"] = str(hvd.local_rank())
except:
    pass

from .trainer import torch_trainer

from src.config import DistributedMode, ComputeMode

# class CTDeepSpeedEngine(deepspeed.DeepSpeedEngine)

class distributed_trainer(torch_trainer):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self, *args, **kwargs):


        torch_trainer.__init__(self, *args, **kwargs)
        # Rely on the base class for most standard parameters, only
        # search for parameters relevant for distributed computing here


        # Put the IO rank as the last rank in the COMM, since rank 0 does tf saves

        if self.args.framework.distributed_mode == DistributedMode.horovod:
            # if self.args.run.compute_mode == "GPU":
                # os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())
            self._rank            = hvd.rank()
            self._local_rank      = hvd.local_rank()
            self._size            = hvd.size()
        else:

            # In the exec.py file, I call a script that sets MPI
            # variables in the environment for every rank.  So a lot of this
            # is simpler than it used to be.
            import torch.distributed as dist
            from torch.nn.parallel import DistributedDataParallel as DDP

            rank       = int(os.environ['RANK'])
            local_rank = int(os.environ['LOCAL_RANK'])
            size       = int(os.environ['WORLD_SIZE'])

            # os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)

            self._rank = rank
            self._size = size
            self._local_rank = local_rank

            # What backend?  nccl on GPU, gloo on CPU
            if self.args.run.compute_mode == ComputeMode.XPU:
                # import torch_ccl
                backend = 'ccl'
            elif self.args.run.compute_mode == ComputeMode.CUDA:
                if self.args.framework.oversubscribe > 1:
                    backend = 'gloo'
                else:
                    backend = 'nccl'
            elif self.args.run.compute_mode == ComputeMode.CPU: backend = 'gloo'

            # init_method = 'file:///home/cadams/ddp_init/ddp_init.txt'
            init_method = 'env://'

            torch.distributed.init_process_group(
                backend     = backend,
                init_method = init_method,
                world_size  = size,
                rank        = rank,
                timeout     = datetime.timedelta(seconds=120)
            )


    def save_model(self):

        if self._rank == 0:
            torch_trainer.save_model(self)

    def default_device_context(self):

        # Convert the input data to torch tensors
        if self.args.run.compute_mode == ComputeMode.CUDA:
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                # Then, it's manually set, use it
                return torch.cuda.device(0)
            else:
                return torch.cuda.device(int(self._local_rank))
        elif self.args.run.compute_mode == ComputeMode.XPU:
            # return contextlib.nullcontext
            try:
                return ipex.xpu.device(int(self._local_rank))
            except:
                pass
            return contextlib.nullcontext
        elif self.args.run.compute_mode == ComputeMode.DPCPP:
            return contextlib.nullcontext
            # device = torch.device("dpcpp")
        else:
            return contextlib.nullcontext
            # device = torch.device('cpu')

    def barrier(self):
        MPI.COMM_WORLD.Barrier()

    def default_device(self):

        if self.args.run.compute_mode == ComputeMode.CUDA:
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                # Then, it's manually set, use it
                return torch.device("cuda:0")
            else:
                return torch.device(f"cuda:{self._local_rank}")
        elif self.args.run.compute_mode == ComputeMode.XPU:
            device = torch.device(f"xpu:{self._local_rank}")
        elif self.args.run.compute_mode == ComputeMode.DPCPP:
            device = torch.device("dpcpp")
        else:
            device = torch.device('cpu')
        return device


    def init_optimizer(self):

        # This takes the base optimizer (self.opt) and replaces
        # it with a distributed version

        torch_trainer.init_optimizer(self)

        if self.args.framework.distributed_mode == DistributedMode.horovod:
            self.opt = hvd.DistributedOptimizer(self.opt, named_parameters=self._net.named_parameters())
            self.opt.param_groups[0]['capturable'] = True
        # self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, self.lr_calculator, last_epoch=-1)


    def init_saver(self):
        if self._rank == 0:
            torch_trainer.init_saver(self)
        else:
            self._saver = None
            self._aux_saver = None


    def print_network_info(self, verbose=False):
        if self._rank == 0:
            torch_trainer.print_network_info(self, verbose)
        return

    def restore_model(self):

        # Load on rank 0:
        if self._rank == 0:
            state = self.load_state_from_file()
        else:
            state = None

        # Restore the weights on rank 0:
        if state is not None and self._rank == 0:
            self.restore_state(state)


        # Broadcast from rank 0 to sync weights
        if self.args.framework.distributed_mode == DistributedMode.horovod:

            # Broadcast the global step:
            self._global_step = hvd.broadcast_object(self._global_step, root_rank = 0)

            # Broadcast the state of the model:
            hvd.broadcast_parameters(self._net.state_dict(), root_rank = 0)

            # Broadcast the optimizer state:
            hvd.broadcast_optimizer_state(self.opt, root_rank = 0)

            # Horovod doesn't actually move the optimizer onto a GPU:
            if self.args.run.compute_mode == ComputeMode.CUDA:
                for state in self.opt.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()



            # Broadcast the LR Schedule state:
            state_dict = hvd.broadcast_object(self.lr_scheduler.state_dict(), root_rank = 0)

        elif self.args.framework.distributed_mode == DistributedMode.DDP:

            devices = None
            if self.args.run.compute_mode == ComputeMode.XPU:
                devices = ["xpu:{}".format(self._local_rank)]
                self._net.to(devices[0])
            elif self.args.run.compute_mode == ComputeMode.CUDA:
                self._net.cuda()

            # print(self._net.parameters)

            self._net = torch.nn.parallel.DistributedDataParallel(self._net, device_ids=devices, find_unused_parameters=False)

            # print(self._net.parameters)

            self._global_step = MPI.COMM_WORLD.bcast(self._global_step, root=0)
            if self.is_training():
                state_dict = MPI.COMM_WORLD.bcast(self.lr_scheduler.state_dict(), root=0)

        elif self.args.framework.distributed_mode == DistributedMode.DeepSpeed:

            model_engine, optimizer

        # Load the state dict:
        if self.is_training():
            self.lr_scheduler.load_state_dict(state_dict)

    def summary(self, metrics, saver=""):

        if self._rank == 0:
            torch_trainer.summary(self, metrics, saver)
        return

    def summary_images(self, logits_image, labels_image, saver=""):
        if self._rank == 0:
            torch_trainer.summary_images(self, logits_image, labels_image, saver)
        return

    def _compute_metrics(self, logits, minibatch_data, loss_dict, batch_reduce=True):

        # This function calls the parent function which computes local metrics.
        # Then, it performs an all reduce on all metrics:
        metrics = torch_trainer._compute_metrics(self, logits, minibatch_data, loss_dict, batch_reduce)

        if batch_reduce:
            # Only perform the reduction if already reduced over the batch:
            if self.args.framework.distributed_mode == DistributedMode.horovod:
                for key in metrics:
                    metrics[key] = hvd.allreduce(metrics[key], name = key)
            elif self.args.framework.distributed_mode == DistributedMode.DDP:
                with self.default_device_context():
                    for key in metrics:
                        torch.distributed.all_reduce(metrics[key])
                        metrics[key] /= self._size

        return metrics




    def log(self, metrics, log_keys, saver):

        if self._rank == 0:
            torch_trainer.log(self, metrics, log_keys, saver)
