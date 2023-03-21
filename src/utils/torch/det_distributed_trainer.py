
import torch
import torch.distributed as dist
from .trainer import torch_trainer
import numpy
import time
import datetime
from src.config import ComputeMode, Precision, ConvMode, ModeKind, DataFormatKind

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

        # need this to report hpo search op.report_completed()
        self._last_known_val_metrics = None


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
        #logger.info(f"Setting default_device_context to {self._local_rank} for rank {self._local_rank}")
        return torch.cuda.device(int(self._local_rank))


    def default_device(self):

        #logger.info(f"Get default_device_context to {self._local_rank} for rank {self._local_rank}")
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

    def batch_process(self):

        # Run iterations
        for op in self.determined_context.searcher.operations():
            start = time.time()
            post_one_time = None
            post_two_time = None

            times = []
            # This is the 'master' function, so it controls a lot
            self.profiling_index = 0
            self.args.run.iterations = op.length
            print (f'setting self.args.run.iterations from searcher to {self.args.run.iterations}')
            for self._iteration in range(self.args.run.iterations):

                # Resize the profiling array if needed:
                if self.profiling_index > len(self.profiling_array) - 1:
                    # Add 500 more rows:
                    self.profiling_array.resize((self.profiling_index + 500))

                self.profiling_array[self.profiling_index]["i"] = self._iteration
                self.profiling_array[self.profiling_index]["start"] = self.now()

                with self.timing_context("iteration"):
                    iteration_start = time.time()
                    if self.is_training() and self._iteration >= self.args.run.iterations:

                        logger.info('Finished training (iteration %d)' % self._iteration)
                        self.checkpoint()
                        break


                    if self.is_training():
                        with self.timing_context("train"):
                            self.train_step()
                            if self._rank == 0: #report only from rank 0
                                op.report_progress(self._iteration)
                        with self.timing_context("val"):
                            self.val_step(op)
                        with self.timing_context("checkpoint"):
                            if self.checkpoint() == 1:
                                logger.info(f'Exiting training at iteration {self._iteration} as pre-emption signal recieved')
                    else:
                        self.ana_step()

                    if post_one_time is None:
                        post_one_time = time.time()
                    elif post_two_time is None:
                        post_two_time = time.time()
                    times.append(time.time() - iteration_start)

                            # check for pre-emption and break 
                    if self.determined_context.preempt.should_preempt():
                        logger.info(f'pre-emption signal received at iteration {self._iteration} - exiting training loop')
                        break

                self.profiling_index += 1
            
            #report to operation completed
            if self._rank == 0: #report only from rank 0
                searcher_conf = self.determined_info.trial._config["searcher"]
                det_searcher_metric_name = searcher_conf["metric"]
                op.report_completed(self._last_known_val_metrics[det_searcher_metric_name])

            self.close_savers()

            self.write_profiling_info()

            end = time.time()

            if self.args.data.synthetic and self.args.run.distributed:
                try:
                    total_images_per_batch = self.args.run.minibatch_size * self._size
                except:
                    total_images_per_batch = self.args.run.minibatch_size
            else:
                total_images_per_batch = self.args.run.minibatch_size


            logger.info(f"Total time to batch_process: {end - start:.4f}")
            if post_one_time is not None:
                throughput = (self.args.run.iterations - 1) * total_images_per_batch
                throughput /= (end - post_one_time)
                logger.info("Total time to batch process except first iteration: "
                            f"{end - post_one_time:.4f}"
                            f", throughput: {throughput:.4f}")
            if post_two_time is not None:
                throughput = (self.args.run.iterations - 2) * total_images_per_batch
                throughput /= (end - post_two_time)
                logger.info("Total time to batch process except first two iterations: "
                            f"{end - post_two_time:.4f}"
                            f", throughput: {throughput:.4f}")
            if len(times) > 40:
                throughput = (40) * total_images_per_batch
                throughput /= (numpy.sum(times[-40:]))
                logger.info("Total time to batch process last 40 iterations: "
                            f"{numpy.sum(times[-40:]):.4f}"
                            f", throughput: {throughput:.4f}" )

    def val_step(self, op):
        # First, validation only occurs on training:
        if not self.is_training(): return

        if self.args.data.synthetic: return
        # Second, validation can not occur without a validation dataloader.
        if self.args.data.aux_file == "": return

        # perform a validation step
        # Validation steps can optionally accumulate over several minibatches, to
        # fit onto a gpu or other accelerator
        if (self._global_step != 0 and self._global_step % self.args.run.aux_iterations == 0) or (self.args.run.iterations == op.length):

            self._net.eval()
            if self.args.run.compute_mode == ComputeMode.CPU:
                # Quantization not supported on CUDA
                val_net = torch.quantization.convert(self._net)
            else:
                val_net = self._net
            # Fetch the next batch of data with larcv
            # (Make sure to pull from the validation set)
            io_start_time = datetime.datetime.now()
            with self.timing_context("io"):
                minibatch_data = self.larcv_fetcher.fetch_next_batch('aux', force_pop = True)
            io_end_time = datetime.datetime.now()

            # if mixed precision, and cuda, use autocast:
            if self.args.run.precision == Precision.mixed and self.args.run.compute_mode == ComputeMode.GPU:
                with torch.cuda.amp.autocast():
                    logits_image, labels_image = self.forward_pass(minibatch_data, net=val_net)
            else:
                logits_image, labels_image = self.forward_pass(minibatch_data, net=val_net)

            # Compute the loss based on the logits
            loss = self.loss_calculator(labels_image, logits_image)

            # Compute any necessary metrics:
            metrics = self._compute_metrics(logits_image, labels_image, loss)

            self.log(metrics, saver="test")
            self.summary(metrics, saver="test")
            self.summary_images(logits_image, labels_image, saver="test")

            return

    def summary(self, metrics, saver=""):

        if self._rank == 0:
            torch_trainer.summary(self, metrics, saver)
            if saver =='train':
                self.determined_context.train.report_training_metrics(
                    steps_completed=self._global_step+1,
                    metrics={k: metrics[k].item() if isinstance(metrics[k], torch.Tensor) else metrics[k] for k in metrics}
                )
                # determined needs validation metric to complete the trail and is also needed in HPO for search space exploration
                # control flow for synthetic data is to avoid validation loop so the hack
                if self.args.data.synthetic:
                    self._last_known_val_metrics = {k: metrics[k].item() if isinstance(metrics[k], torch.Tensor) else metrics[k] for k in metrics}
                    #logger.info(f"hack when using synthetic data to use training last known metric: {self._last_known_val_metrics}")
            if saver == 'test':
                self._last_known_val_metrics = {k: metrics[k].item() if isinstance(metrics[k], torch.Tensor) else metrics[k] for k in metrics}
                self.determined_context.train.report_validation_metrics(
                    steps_completed=self._global_step,
                    metrics=self._last_known_val_metrics
                )
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