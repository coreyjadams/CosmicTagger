
import torch
import torch.distributed as dist
from .trainer import torch_trainer

import logging
logger = logging.getLogger()
logger.propogate = False


# want to use mpi for bcast non tensors
from mpi4py import MPI
comm = MPI.COMM_WORLD

import pathlib
from determined import core


class det_distributed_trainer(torch_trainer):

    def __init__(self, determined_context, determined_info, args):

        torch_trainer.__init__(self, args)


        self.determined_context = determined_context
        self.determined_info = determined_info
        self._size = int(self.determined_context.distributed.get_size())
        self._rank = int(self.determined_context.distributed.get_rank())
        self._local_rank = int(self.determined_context.distributed.get_local_rank())


        # Initialize torch distributed
        dist.init_process_group(backend="NCCL", world_size=self._size, rank=self._rank)

    def checkpoint(self):
        if self._global_step % self.args.mode.checkpoint_iteration == 0 and self._global_step != 0:
            # Save a checkpoint, but don't do it on the first pass
            self.save_model()

    def save_model(self):
        if self._rank == 0:
            checkpoint_metadata = {"steps_completed": self._global_step}
            with self.determined_context.checkpoint.store_path(checkpoint_metadata) as (checkpoint_directory, uuid):
                state_dict = {
                    'global_step' : self._global_step,
                    'state_dict'  : self._net.state_dict(),
                    'optimizer'   : self._opt.state_dict(),
                    'scheduler'   : self.lr_scheduler.state_dict(),
                }
                ckpt_file = checkpoint_directory.joinpath('state.ckpt')
                torch.save(state_dict, ckpt_file)

    def load_state(self):
        if self.determined_info.latest_checkpoint is not None:
            with self.determined_context.checkpoint.restore_path(self.determined_info.latest_checkpoint, download_mode=core.DownloadMode.NoSharedDownload) as path:
                checkpoint_directory = pathlib.Path(path)
                ckpt_file = checkpoint_directory.joinpath('state.ckpt')
                state = torch.load(ckpt_file)
                print (f'loaded state from {ckpt_file}')
                return state


    def default_device_context(self):
        logger.info(f"Setting default_device_context to {self._local_rank} for rank {self._local_rank}")
        return torch.cuda.device(int(self._local_rank))


    def default_device(self):

        logger.info(f"Get default_device_context to {self._local_rank} for rank {self._local_rank}")
        return torch.device(f"cuda:{self._local_rank}")

    def barrier(self):
        MPI.COMM_WORLD.Barrier()

    def init_optimizer(self):

        # This takes the base optimizer (self._opt) and replaces
        # it with a distributed version

        torch_trainer.init_optimizer(self)

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self._opt, self.lr_calculator, last_epoch=-1)


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
            state = self.load_state()
        else:
            state = None

        # Restore the weights on rank 0:
        if state is not None and self._rank == 0:
            self.restore_state(state)


        # Broadcast from rank 0 to sync weights

        devices = None
        self._net.cuda()

        # print(self._net.parameters)

        self._net = torch.nn.parallel.DistributedDataParallel(self._net, device_ids=devices, find_unused_parameters=False)

        # print(self._net.parameters)

        self._global_step = MPI.COMM_WORLD.bcast(self._global_step, root=0)

        state_dict = MPI.COMM_WORLD.bcast(self.lr_scheduler.state_dict(), root=0)


        # Load the state dict:
        self.lr_scheduler.load_state_dict(state_dict)

    def val_step(self):
        torch_trainer.val_step(self)

    def summary(self, metrics, saver=""):

        if self._rank == 0:
            torch_trainer.summary(self, metrics, saver)
            if saver =='train':
                self.determined_context.train.report_training_metrics(
                    steps_completed=self._global_step,
                    metrics={k: metrics[k].item() if isinstance(metrics[k], torch.Tensor) else metrics[k] for k in metrics}
                )
            if saver == 'test':
                for k in metrics:
                    print (f'{k} - {metrics[k]}')
                    print (type(metrics[k]))
                #self.determined_context.train.report_validation_metrics(
                 #   steps_completed=self._global_step,
                  #  metrics={k: metrics[k] for k in metrics}
                #)
        return

    def summary_images(self, logits_image, labels_image, saver=""):
        if self._rank == 0:
            torch_trainer.summary_images(self, logits_image, labels_image, saver)
        return

    def _compute_metrics(self, logits, minibatch_data, loss):

        # This function calls the parent function which computes local metrics.
        # Then, it performs an all reduce on all metrics:
        metrics = torch_trainer._compute_metrics(self, logits, minibatch_data, loss)

        for key in metrics:
            torch.distributed.all_reduce(metrics[key])
            metrics[key] /= self._size

        return metrics


    def log(self, metrics, saver=""):
        if self._rank == 0:
            torch_trainer.log(self, metrics, saver)