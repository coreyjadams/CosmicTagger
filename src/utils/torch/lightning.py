import os
import sys
import time
import tempfile
import datetime
from collections import OrderedDict

from src.utils import logging
logger = logging.getLogger("CosmicTagger")
# logger.propogate = False

import numpy


import torch
import pytorch_lightning as pl

torch.autograd.set_detect_anomaly(True)

try:
    import ipex
except:
    pass




torch.manual_seed(0)

# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


# from torch.jit import trace

from src.networks.torch         import LossCalculator, AccuracyCalculator, predict_vertex


from src.config import ComputeMode, Precision, ConvMode, ModeKind, OptimizerKind


class lightning_trainer(pl.LightningModule):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self, args, model, loss_calc,
        acc_calc, lr_scheduler=None, log_keys = [], hparams_keys = [], vertex_meta = None):
        super().__init__()

        self.args         = args
        self.model        = model
        self.lr_scheduler = lr_scheduler
        self.vertex_meta  = vertex_meta
        self.loss_calc    = loss_calc
        self.acc_calc     = acc_calc

        self.log_keys     = log_keys
        self.hparams_keys = hparams_keys

        self.initialize_throughput_parameters()


    def prep_labels(self, minibatch_data):

        # Always have segmentation labels:
        labels_dict = {
            "segmentation" : torch.chunk(minibatch_data['label'].long(), chunks=3, dim=1),
        }

        # Adjust the shape of the labels:
        shape =  labels_dict["segmentation"][0].shape
        labels_dict["segmentation"] = [
            _label.view([shape[0], shape[-2], shape[-1]])
                for _label in labels_dict["segmentation"]
        ]

        # Add event id and vertex as needed:
        if self.args.network.classification.active or self.args.network.vertex.active:
            labels_dict.update({"event_label"  : minibatch_data['event_label']})
        if self.args.network.vertex.active:
            labels_dict.update({"vertex"  : minibatch_data['vertex']})

        return labels_dict

    ####################################################################
    # Starting here, the following functions are purely for benchmarking
    ####################################################################

    def initialize_throughput_parameters(self):
        # Just for benchmarking measurements:
        self.post_one_time   = None
        self.post_two_time   = None
        self.iteration_start = None
        self.times           = []
        self.start_time      = None
        self.end_time        = None
        self.n_iteration     = 0

    def on_fit_start(self):
        super().on_fit_start()
        self.record_start_time()

    def on_test_start(self):
        super().on_test_start()
        self.record_start_time()

    def record_start_time(self):
        self.start_time = time.time()

    def on_fit_end(self):
        super().on_fit_end()
        self.record_end_time()

    def on_test_end(self):
        super().on_test_end()
        self.record_end_time()

    def record_end_time(self):
        self.end_time = time.time()
        self.throughput_report()

    def on_train_batch_start(self, batch, batch_idx):
        super().on_train_batch_start(batch, batch_idx)
        self.record_iteration_start_time()

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx=0):
        super().on_test_batch_start(batch, batch_idx, dataloader_idx)
        self.record_iteration_start_time()

    def record_iteration_start_time(self):
        self.iteration_start = time.time()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        super().on_train_batch_end(outputs, batch, batch_idx)
        self.record_iteration_end_time()

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        super().on_test_batch_end(outputs, batch, batch_idx, dataloader_idx)
        self.record_iteration_end_time()

    def record_iteration_end_time(self):
        if self.post_one_time is None:
            self.post_one_time = time.time()
        elif self.post_two_time is None:
            self.post_two_time = time.time()
        self.times.append(time.time() - self.iteration_start)
        self.n_iteration += 1

    def throughput_report(self):
        total_images_per_batch = self.args.run.minibatch_size
        if self.args.data.synthetic and self.args.run.distributed:
            total_images_per_batch = self.args.run.minibatch_size * int(os.environ["WORLD_SIZE"])


        # if self.args.mode.name == ModeKind.inference:
        #     self.inference_report()

        logger.info(f"Total time to batch_process: {self.end_time - self.start_time:.4f}")
        if self.post_one_time is not None:
            throughput = (self.n_iteration - 1) * total_images_per_batch
            time       = (self.end_time - self.post_one_time)
            throughput /= time
            logger.info("Total time to batch process except first iteration: "
                        f"{time:.4f}, throughput: {throughput:.4f}")
        if self.post_two_time is not None:
            throughput = (self.n_iteration - 2) * total_images_per_batch
            time       = (self.end_time - self.post_two_time)
            throughput /= time
            logger.info("Total time to batch process except first two iterations: "
                        f"{time:.4f}, throughput: {throughput:.4f}")
        if len(self.times) > 40:
            throughput = (40) * total_images_per_batch
            time       = (numpy.sum(self.times[-40:]))
            throughput /= time
            logger.info("Total time to batch process last 40 iterations: "
                        f"{time:.4f}, throughput: {throughput:.4f}" )


    ####################################################################
    # This concludes the section of functions for benchmarking only
    ####################################################################


    def forward(self, batch):
        network_dict = self.model(batch)
        return network_dict


    def test_step(self, batch, batch_idx):
        network_dict = self(batch['image'])
        prepped_labels = self.prep_labels(batch)

        acc_metrics  = self.calculate_accuracy(network_dict, prepped_labels)

        self.print_log(acc_metrics, mode="test")
        # self.summary(acc_metrics)
        # self.log_dict(acc_metrics)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        network_dict = self(batch["image"])
        prepped_labels = self.prep_labels(batch)

        loss, loss_metrics = self.loss_calc(prepped_labels, network_dict)

        acc_metrics = self.calculate_accuracy(network_dict, prepped_labels)

        acc_metrics.update({
            f"loss/{key}" : loss_metrics[key] for key in loss_metrics
        })


        self.print_log(acc_metrics, mode="train")
        # self.summary(acc_metrics)
        self.log_dict(acc_metrics, sync_dist=True, rank_zero_only=True)
        return loss

    def configure_optimizers(self):
        learning_rate = 1.0
        # learning_rate = self.args.mode.optimizer.learning_rate

        if self.args.mode.optimizer.name == OptimizerKind.rmsprop:
            opt = torch.optim.RMSprop(self.parameters(), learning_rate, eps=1e-6)
        elif self.args.mode.optimizer.name == OptimizerKind.adam:
            opt = torch.optim.Adam(self.parameters(), learning_rate, eps=1e-6, betas=(0.8,0.9))
        elif self.args.mode.optimizer.name == OptimizerKind.adagrad:
            opt = torch.optim.Adagrad(self.parameters(), learning_rate)
        elif self.args.mode.optimizer.name == OptimizerKind.adadelta:
            opt = torch.optim.Adadelta(self.parameters(), learning_rate, eps=1e-6)
        else:
            opt = torch.optim.SGD(self.parameters(), learning_rate)

        lr_fn = lambda x : self.lr_scheduler[x]

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn, last_epoch=-1)

        return [opt],[{"scheduler" : lr_scheduler, "interval": "step"}]

    def print_network_info(self, verbose=False):
        if verbose:
            for name, var in self._net.named_parameters():
                logger.info(f"{name}: {var.shape}")

        logger.info("Total number of trainable parameters in this network: {}".format(self.n_parameters()))


    def n_parameters(self):
        n_trainable_parameters = 0
        for name, var in self._net.named_parameters():
            n_trainable_parameters += numpy.prod(var.shape)
        return n_trainable_parameters

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch


    def store_parameters(self, metrics,):
        ''' Store all the hyperparameters with MLFLow'''
        flattened_dict = self.flatten(self.args)
        hparams_metrics = {}
        if self.args.mode.name == ModeKind.inference:
            return
        for key in self.hparams_keys:
            if key not in metrics: continue
            hparams_metrics[key] = float(metrics[key].float().cpu())
        if hasattr(self, "_aux_saver") and self._aux_saver is not None:
            self._aux_saver.add_hparams(flattened_dict, hparams_metrics, run_name="hparams")
            self._aux_saver.flush()
        return


    def calculate_accuracy(self, network_dict, labels_dict):
        ''' Calculate the accuracy.

            Images received here are not sparse but dense.
            This is to ensure equivalent metrics are computed for sparse and dense networks.

        '''

        # Predict the vertex, if needed:
        if self.args.network.vertex.active:
            network_dict['predicted_vertex'] = predict_vertex(network_dict, self.vertex_meta)

        return self.acc_calc(network_dict, labels_dict)


    def _compute_metrics(self, network_dict, labels_dict, loss_dict):

        with torch.no_grad():
            # Call all of the functions in the metrics dictionary:
            metrics = {}

            if loss_dict is not None:
                for key in loss_dict:
                    metrics[f'loss/{key}'] = loss_dict[key].data
            accuracy = self._calculate_accuracy(network_dict, labels_dict)
            accuracy.update(metrics)

        current_lr = self.optimizers().state_dict()['param_groups'][0]['lr']
        metrics["learning_rate"] = current_lr
        return metrics

    def print_log(self, metrics, mode=""):

        if self.global_step % self.args.mode.logging_iteration == 0:

            self._current_log_time = datetime.datetime.now()

            # Build up a string for logging:
            if self.log_keys != []:
                s = ", ".join(["{0}: {1:.3}".format(key, metrics[key]) for key in self.log_keys])
            else:
                s = ", ".join(["{0}: {1:.3}".format(key, metrics[key]) for key in metrics])

            time_string = []

            if hasattr(self, "_previous_log_time"):
            # try:
                total_images = self.args.run.minibatch_size
                images_per_second = total_images / (self._current_log_time - self._previous_log_time).total_seconds()
                time_string.append("{:.2} Img/s".format(images_per_second))

            if 'io_fetch_time' in metrics.keys():
                time_string.append("{:.2} IOs".format(metrics['io_fetch_time']))

            if 'step_time' in metrics.keys():
                time_string.append("{:.2} (Step)(s)".format(metrics['step_time']))

            if len(time_string) > 0:
                s += " (" + " / ".join(time_string) + ")"

            # except:
            #     pass


            self._previous_log_time = self._current_log_time
            logger.info("{} Step {} metrics: {}".format(mode, self.global_step, s))

    def exit(self): pass

    # def summary(self, metrics):
    #     if self.global_step % self.args.mode.summary_iteration == 0:
    #         saver = self.logger.experiment
    #         for metric in metrics:
    #             name = metric
    #             value = metrics[metric]
    #             if isinstance(value, torch.Tensor):
    #                 # Cast metrics to 32 bit float
    #                 value = value.float()
    #
    #             saver.add_scalar(metric, value, self.global_step)
    #
    #         # try to get the learning rate
    #         try:
    #             current_lr = self.optimizers().state_dict()['param_groups'][0]['lr']
    #             saver.add_scalar("learning_rate", current_lr, self.global_step)
    #         except:
    #             pass
    #         return

    def accumulate_metrics(self, metrics):

        self.inference_metrics['n'] += 1
        for key in metrics:
            if key not in self.inference_metrics:
                self.inference_metrics[key] = metrics[key]
                # self.inference_metrics[f"{key}_sq"] = metrics[key]**2
            else:
                self.inference_metrics[key] += metrics[key]
                # self.inference_metrics[f"{key}_sq"] += metrics[key]**2

    def inference_report(self):
        if not hasattr(self, "inference_metrics"):
            return
        n = self.inference_metrics["n"]
        total_entries = n*self.args.run.minibatch_size
        logger.info(f"Inference report: {n} batches processed for {total_entries} entries.")
        for key in self.inference_metrics:
            if key == 'n' or '_sq' in key: continue
            value = self.inference_metrics[key] / n
            logger.info(f"  {key}: {value:.4f}")



def build_network(args, image_size):

    from src.config import ConvMode

    if args.network.conv_mode == ConvMode.conv_2D and not args.framework.sparse:
        from src.networks.torch.uresnet2D import UResNet
        net = UResNet(args.network, image_size)

    else:
        if args.framework.sparse and args.mode.name != ModeKind.iotest:
            from src.networks.torch.sparseuresnet3D import UResNet3D
        else:
            from src.networks.torch.uresnet3D       import UResNet3D

        net = UResNet3D(args.network, image_size)

    return net

from . data import create_torch_larcv_dataloader
from src.networks.torch import create_vertex_meta

def create_lightning_module(args, datasets, lr_scheduler=None, log_keys = [], hparams_keys = []):

    # Going to build up the lightning module here.

    # Take the first dataset:
    example_ds = next(iter(datasets.values()))

    vertex_meta = create_vertex_meta(args, example_ds.image_meta, example_ds.image_size())

    # Next, create the network:
    network = build_network(args, example_ds.image_size())

    if args.mode.name == ModeKind.train:
        if args.network.classification.active:
            weight = torch.tensor([0.16, 0.1666, 0.16666, 0.5])
            loss_calc = LossCalculator(args, weight=weight)
        else:
            loss_calc = LossCalculator(args)
        acc_calc = AccuracyCalculator(args)
    else:
        loss_calc = None
        acc_calc  = AccuracyCalculator(args)


    model = lightning_trainer(args, network, loss_calc,
        acc_calc, lr_scheduler,
        log_keys     = log_keys,
        hparams_keys = hparams_keys,
        vertex_meta  = vertex_meta)
    return model

def train(args, lightning_model, datasets, max_epochs=None, max_steps=None):

    from src.config import Precision

    # Map the precision to lightning args:
    if args.run.precision == Precision.mixed:
        precision = 16
    elif args.run.precision == Precision.bfloat16:
        precision = "bf16"
    else:
        precision = 32

    # Map the profiling to lightning args:
    if args.run.profile:
        profiler = "advanced"
    else:
        profiler  = None

    if args.run.distributed:
        from lightning_fabric.plugins.environments import MPIEnvironment
        environment = MPIEnvironment()
    else:
        from lightning_fabric.plugins.environments import LightningEnvironment
        environment = LightningEnvironment()


    # Distributed strategy:
    if args.run.distributed:
        from src.config import DistributedMode
        if args.framework.distributed_mode == DistributedMode.horovod:
            from pytorch_lightning.strategies import DataParallelStrategy
            strategy = DataParallelStrategy(
                cluster_environment = environment,

            )
        elif args.framework.distributed_mode == DistributedMode.DDP:
            from pytorch_lightning.strategies import DDPStrategy
            strategy = DDPStrategy(
                cluster_environment = environment,
            )
        elif False:
            from pytorch_lightning.strategies import DDPFullyShardedStrategy
            strategy = DDPFullyShardedStrategy(cluster_environment = environment,)
        elif args.framework.distributed_mode == DistributedMode.deepspeed:
            from pytorch_lightning.strategies import DeepSpeedStrategy
            strategy = DeepSpeedStrategy(
                zero_optimization   = True,
                stage               = 3,
                sub_group_size      = 100000000000,
                offload_optimizer   = True,
                cluster_environment = environment,
            )
        elif args.framework.distributed_mode == DistributedMode.sharded:
            from pytorch_lightning.strategies import DDPShardedStrategy
            strategy = DDPShardedStrategy(
                cluster_environment = environment,
            )

        devices   = int(os.environ['LOCAL_SIZE'])
        num_nodes = int(os.environ['N_NODES'])
        plugins   = []
        # if args.run.compute_mode == ComputeMode.CUDA:
        #     os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['LOCAL_RANK']
        #     devices=1
    else:
        plugins   = []
        strategy  = None
        devices   = 1
        num_nodes = 1

    # Move data to the device in the data loader:
    if args.run.compute_mode == ComputeMode.CUDA:
        target_device = torch.device(environment.local_rank())
        # from lightning.pytorch.accelerators import find_usable_cuda_devices
        # lightning_devices = find_usable_cuda_devices(int(os.environ['LOCAL_RANK']))
        # print(lightning_devices)
    else:
        target_device = None

    # Turn the datasets into dataloaders:
    for key in datasets.keys():
        datasets[key] = create_torch_larcv_dataloader(
            datasets[key], args.run.minibatch_size, device=target_device)


    # Configure the logger:
    from pytorch_lightning.loggers import TensorBoardLogger

    tb_logger = TensorBoardLogger(args.output_dir + "/train/")

    # Hooks specific to training:
    if args.mode.name == ModeKind.train:
        accumulate = args.mode.optimizer.gradient_accumulation
    else:
        # Limit the prediction steps, in this case, to what is specified:
        if max_steps is None:
            max_steps = len(dataloaders["test"]) * max_epochs
        accumulate = None

    trainer = pl.Trainer(
        accelerator             = args.run.compute_mode.name.lower(),
        # num_nodes               = num_nodes,
        default_root_dir        = args.output_dir,
        precision               = precision,
        profiler                = profiler,
        strategy                = strategy,
        enable_progress_bar     = False,
        replace_sampler_ddp     = False,
        logger                  = tb_logger,
        log_every_n_steps       = 1,
        max_epochs              = max_epochs,
        max_steps               = max_steps,
        plugins                 = plugins,
        # benchmark               = True,
        accumulate_grad_batches = accumulate,
        limit_test_batches      = max_steps
    )



    if args.mode.name == ModeKind.train:
        trainer.fit(
            lightning_model,
            train_dataloaders=datasets["train"],
        )
    elif args.mode.name == ModeKind.inference:
        trainer.test(
            lightning_model,
            dataloaders=datasets["test"],
        )
