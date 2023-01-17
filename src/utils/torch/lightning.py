import os
import sys
import time
import tempfile
import datetime
from collections import OrderedDict

import logging
logger = logging.getLogger("CosmicTagger")
logger.propogate = False

import numpy


import torch
import pytorch_lightning as pl
logging.basicConfig(level=logging.INFO)

torch.autograd.set_detect_anomaly(True)

try:
    import ipex
except:
    pass




torch.manual_seed(0)

# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


# from torch.jit import trace

from src.utils.core.trainercore import trainercore
from src.networks.torch         import LossCalculator, AccuracyCalculator, predict_vertex


from src.config import ComputeMode, Precision, ConvMode, ModeKind, OptimizerKind

from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities import rank_zero_only

# class CTLogger(Logger):

#     @property
#     def name(self):
#         return "CosmicTagger"

#     @property
#     def version(self):
#         # Return the experiment version, int or str.
#         return "0.1"

#     @rank_zero_only
#     def log_hyperparams(self, params):
#         # params is an argparse.Namespace
#         # your code to record hyperparameters goes here
#         pass

#     @rank_zero_only
#     def log_metrics(self, metrics, step):
#         # metrics is a dictionary of metric names and values
#         # your code to record metrics goes here
#         pass

#     @rank_zero_only
#     def save(self):
#         # Optional. Any code necessary to save logger data goes here
#         pass

#     @rank_zero_only
#     def finalize(self, status):
#         # Optional. Any code that needs to be run after training
#         # finishes goes here
#         pass




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

    def forward(self, batch):

        network_dict = self.model(batch)

        return network_dict


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.


        network_dict = self(batch["image"])
        prepped_labels = self.prep_labels(batch)

        loss, loss_metrics = self.loss_calc(prepped_labels, network_dict)

        acc_metrics = self.calculate_accuracy(network_dict, prepped_labels)

        acc_metrics.update({
            f"loss/{key}" : loss_metrics[key] for key in loss_metrics
        })

        # print(loggers)

        self.print_log(acc_metrics, mode="train")
        self.summary(acc_metrics)
        self.log_dict(acc_metrics)
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

        return accuracy

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

    def summary(self, metrics):
        if self.global_step % self.args.mode.summary_iteration == 0:
            saver = self.logger.experiment
            for metric in metrics:
                name = metric
                value = metrics[metric]
                if isinstance(value, torch.Tensor):
                    # Cast metrics to 32 bit float
                    value = value.float()

                saver.add_scalar(metric, value, self.global_step)

            # try to get the learning rate
            try:
                current_lr = self.optimizers().state_dict()['param_groups'][0]['lr']
                saver.add_scalar("learning_rate", current_lr, self.global_step)
            except:
                pass
            return


    def summary_images(self, logits_image, labels_image, saver=""):

        # if self.global_step % 1 * self.args.mode.summary_iteration == 0:
        if self.global_step % 25 * self.args.mode.summary_iteration == 0 and not self.args.mode.no_summary_images:

            for plane in range(3):
                val, prediction = torch.max(logits_image[plane][0], dim=0)
                # This is a reshape to add the required channels dimension:
                prediction = prediction.view(
                    [1, prediction.shape[-2], prediction.shape[-1]]
                    ).float()


                labels = labels_image[plane][0]
                labels =labels.view(
                    [1,labels.shape[-2],labels.shape[-1]]
                    ).float()

                # The images are in the format (Plane, H, W)
                # Need to transpose the last two dims in order to meet the (CHW) ordering
                # of tensorboardX


                # Values get mapped to gray scale, so put them in the range (0,1)
                labels[labels == 1] = 0.5
                labels[labels == 2] = 1.0


                prediction[prediction == 1] = 0.5
                prediction[prediction == 2] = 1.0


                if saver == "test":
                    self._aux_saver.add_image("prediction/plane_{}".format(plane),
                        prediction, self._global_step)
                    self._aux_saver.add_image("label/plane_{}".format(plane),
                        labels, self._global_step)

                else:
                    self._saver.add_image("prediction/plane_{}".format(plane),
                        prediction, self._global_step)
                    self._saver.add_image("label/plane_{}".format(plane),
                        labels, self._global_step)

        return

    def graph_summary(self):

        if self._global_step % 1 * self.args.mode.summary_iteration == 0:
        # if self._global_step % 25 * self.args.mode.summary_iteration == 0 and not self.args.mode.no_summary_images:
            for name, param in self._net.named_parameters():

                self._saver.add_histogram(f"{name}/weights",
                    param, self._global_step)
                self._saver.add_histogram(f"{name}/grad",
                    param.grad, self._global_step)
                # self._saver.add_histogram(f"{name}/ratio",
                #     param.grad / param, self._global_step)

        return


    def increment_global_step(self):

        self._global_step += 1

        self.on_step_end()


    def exit(self):
        pass

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

def create_vertex_meta(args, image_meta, image_shape):

    # To predict the vertex, we first figure out the size of each bounding box:
    # Vertex comes out with shape :
    # [batch_size, channels, max_boxes, 2*ndim (so 4, in this case)]
    vertex_depth = args.network.depth - args.network.vertex.depth
    vertex_output_space = tuple(d // 2**vertex_depth  for d in image_shape )
    anchor_size = image_meta['size'] / vertex_output_space

    origin = image_meta['origin']

    return {
        "origin"              : origin,
        "vertex_output_space" : vertex_output_space,
        "anchor_size"         : anchor_size
    }

from . data import create_torch_larcv_dataloader

def create_lightning_module(args, datasets, lr_scheduler=None, log_keys = [], hparams_keys = []):

    # Going to build up the lightning module here.

    # Take the first dataset:
    example_ds = next(iter(datasets.values()))

    vertex_meta = create_vertex_meta(args, example_ds.image_meta, example_ds.image_size())

    # Turn the datasets into dataloaders:
    for key in datasets.keys():
        datasets[key] = create_torch_larcv_dataloader(
            datasets[key], args.run.minibatch_size)

    # Next, create the network:
    network = build_network(args, example_ds.image_size())

    if args.network.classification.active:
        weight = torch.tensor([0.16, 0.1666, 0.16666, 0.5])
        loss_calc = LossCalculator(args, weight=weight)
    else:
        loss_calc = LossCalculator(args)
    acc_calc = AccuracyCalculator(args)


    model = lightning_trainer(args, network, loss_calc,
        acc_calc, lr_scheduler,
        log_keys     = log_keys,
        hparams_keys = hparams_keys,
        vertex_meta  = vertex_meta)
    return model

def train(args, lightning_model, datasets):

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

    # Distributed strategy:
    if args.run.distributed:
        from src.config import DistributedMode
        if args.framework.distributed_mode == DistributedMode.horovod:
            strategy = "horovod"
        elif args.framework.distributed_mode == DistributedMode.DDP:
            strategy = "ddp"
        elif args.framework.distributed_mode == DistributedMode.deepspeed:
            strategy = "deepspeed"
    else:
        strategy = None

    # Configure the logger:
    from pytorch_lightning.loggers import TensorBoardLogger

    tb_logger = TensorBoardLogger(args.output_dir + "/train/")

    print(len(datasets["train"]))

    trainer = pl.Trainer(
        accelerator             = args.run.compute_mode.name.lower(),
        devices                 = 1,
        auto_select_gpus        = True,
        default_root_dir        = args.output_dir,
        precision               = precision,
        profiler                = profiler,
        strategy                = strategy,
        enable_progress_bar     = False,
        replace_sampler_ddp     = True,
        logger                  = tb_logger,
        max_epochs              = 2,
        # benchmark               = True,
        accumulate_grad_batches = args.mode.optimizer.gradient_accumulation,
    )

    trainer.fit(
        lightning_model,
        train_dataloaders=datasets["train"],
    )
