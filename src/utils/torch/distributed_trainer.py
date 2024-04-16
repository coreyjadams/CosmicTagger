import os
import sys
import time, datetime
from collections import OrderedDict

import contextlib

# from hanging_threads import start_monitoring
# monitoring_thread = start_monitoring()

import numpy

import torch

# Always want mpi, but horovod is imported below.
from mpi4py import MPI
comm = MPI.COMM_WORLD

try:
    import intel_extension_for_pytorch as ipex
except:
    pass



try:
    import horovod.torch as hvd
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
            hvd.init()
            
            # if self.args.run.compute_mode == "GPU":
                # os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())
            self.rank            = hvd.rank()
            # if self.rank == 0:
                # monitoring_thread = start_monitoring()
            
            self.local_rank      = hvd.local_rank()
            self.size            = hvd.size()
        else:

            # In the exec.py file, I call a script that sets MPI
            # variables in the environment for every rank.  So a lot of this
            # is simpler than it used to be.
            from torch.nn.parallel import DistributedDataParallel as DDP

            rank       = int(os.environ['RANK'])
            # if rank == 0:
                # monitoring_thread = start_monitoring()
            
            local_rank = int(os.environ['LOCAL_RANK'])
            size       = int(os.environ['WORLD_SIZE'])

            # os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)

            self.rank = rank
            self.size = size
            self.local_rank = local_rank


            # What backend?  nccl on GPU, gloo on CPU
            if self.args.run.compute_mode == ComputeMode.XPU:
                import oneccl_bindings_for_pytorch
                backend = 'ccl'
            elif self.args.run.compute_mode == ComputeMode.CUDA:
                if self.args.framework.oversubscribe > 1:
                    backend = 'gloo'
                else:
                    backend = 'nccl'
            elif self.args.run.compute_mode == ComputeMode.CPU: backend = 'gloo'

            init_method = 'env://'


            torch.distributed.init_process_group(
                backend     = backend,
                init_method = init_method,
                world_size  = size,
                rank        = rank,
                timeout     = datetime.timedelta(seconds=120)
            )
            # print("Call DDP barrier", flush=True)
            # torch.distributed.barrier()

            # print("DDP Init done, call all_reduce", flush=True)
            # # Including a dummy barrier:    
            # dummy_tensor = torch.ones((100,), device=self.default_device())
            # torch.distributed.all_reduce(dummy_tensor)

    def save_model(self):

        if self.rank == 0:
            torch_trainer.save_model(self)

    def default_device_context(self):

        # Convert the input data to torch tensors
        if self.args.run.compute_mode == ComputeMode.CUDA:
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                # Then, it's manually set, use it
                return torch.cuda.device(0)
            else:
                return torch.cuda.device(int(self.local_rank))
        elif self.args.run.compute_mode == ComputeMode.XPU:
            # return contextlib.nullcontext
            try:
                return ipex.xpu.device(int(self.local_rank))
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
                return torch.device(f"cuda:{self.local_rank}")
        elif self.args.run.compute_mode == ComputeMode.XPU:
            device = torch.device(f"xpu:{self.local_rank}")
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
            self.opt = hvd.DistributedOptimizer(self.opt, 
                         named_parameters = self._net.named_parameters(), 
                         num_groups       = self.args.run.horovod_num_groups)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, self.lr_calculator, last_epoch=-1)


    def init_saver(self):
        if self.rank == 0:
            torch_trainer.init_saver(self)
        else:
            self._saver = None
            self._aux_saver = None


    def print_network_info(self, verbose=False):
        if self.rank == 0:
            torch_trainer.print_network_info(self, verbose)
        return

    def restore_model(self):

        # Load on rank 0:
        if self.rank == 0:
            state = self.load_state_from_file()
        else:
            state = None


        # Restore the weights on rank 0:
        if state is not None and self.rank == 0:
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
                devices = ["xpu:{}".format(self.local_rank)]
                self._net.to(devices[0])
            elif self.args.run.compute_mode == ComputeMode.CUDA:
                self._net.cuda()


            self._net = torch.nn.parallel.DistributedDataParallel(self._net, device_ids=devices, broadcast_buffers=self.args.run.broadcast_buffers, find_unused_parameters=False)


            self._global_step = MPI.COMM_WORLD.bcast(self._global_step, root=0)
            if self.is_training():
                state_dict = MPI.COMM_WORLD.bcast(self.lr_scheduler.state_dict(), root=0)

        elif self.args.framework.distributed_mode == DistributedMode.DeepSpeed:

            model_engine, optimizer

        # Load the state dict:
        if self.is_training():
            self.lr_scheduler.load_state_dict(state_dict)

    def summary(self, metrics, saver=""):

        if self.rank == 0:
            torch_trainer.summary(self, metrics, saver)
        return

    def summary_images(self, logits_image, labels_image, saver=""):
        if self.rank == 0:
            torch_trainer.summary_images(self, logits_image, labels_image, saver)
        return



    def stack_tensors(self, input_tensor_dict):
        # Assuming we have scalar metrics

        # shapes = { key : input_tensor_dict[key].shape for key in input_tensor_dict.keys()}

        flat_tensors = [ torch.reshape(input_tensor_dict[key], (-1,)) for key in input_tensor_dict.keys() ]

        stacked_tensors = torch.stack(flat_tensors)
        return stacked_tensors # , shapes

    def split_metrics(self, input_stacked_tensor, keys):

        """
        Implicitly assuming that the metrics are all scalars
        """

        n_splits = input_stacked_tensor.shape[0]

        split_metrics = torch.chunk(input_stacked_tensor, chunks = n_splits)

        metrics_dict = {
            key : torch.reshape(m, ()) for key, m in zip(keys, split_metrics)
        }

        return metrics_dict


    def _compute_metrics(self, logits, minibatch_data, loss_dict, batch_reduce=True):

        # This function calls the parent function which computes local metrics.
        # Then, it performs an all reduce on all metrics:
        metrics = torch_trainer._compute_metrics(self, logits, minibatch_data, loss_dict, batch_reduce)

        if batch_reduce:
            # Only perform the reduction if already reduced over the batch:
            stacked_metrics = self.stack_tensors(metrics)

            if self.args.framework.distributed_mode == DistributedMode.horovod:
                stacked_metrics = hvd.allreduce(stacked_metrics, name = "key")
            elif self.args.framework.distributed_mode == DistributedMode.DDP:
                # For an unknown reason, on Polaris, this puts extra stuff on device 0
                # But, ONLY If the validation data is being used.
                torch.distributed.all_reduce(stacked_metrics)

                stacked_metrics /= self.size

            metrics = self.split_metrics(stacked_metrics, metrics.keys())



        return metrics




    def log(self, metrics, log_keys, saver):

        if self.rank == 0:
            torch_trainer.log(self, metrics, log_keys, saver)
