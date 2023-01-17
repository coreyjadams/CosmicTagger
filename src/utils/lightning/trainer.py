import os
import sys
import time
import tempfile
from collections import OrderedDict

import logging
logger = logging.getLogger("CosmicTagger")
logger.propogate = False


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

from src.utils.core.trainercore import trainercore
from src.networks.torch         import LossCalculator, AccuracyCalculator


from src.config import ComputeMode, Precision, ConvMode, ModeKind, OptimizerKind

class lightning_trainer(pl.LightningModule):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self, args, model, loss_calc, acc_calc, lr_scheduler=None, vertex_meta = None):
        super().__init__()

        self.args  = args
        self.model = model
        self.lr_scheduler = lr_scheduler
        self.vertex_meta = vertex_meta

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

    def forward(self, batch):

        print(batch.shape)

        network_dict = self.model(batch)

        return network_dict


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.

        network_dict = self(batch["image"])
        prepped_labels = self.prep_labels(batch)

        loss = self.loss_calc(prepped_labels, network_dict)
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
        
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, self.lr_scheduler, last_epoch=-1)

        return {
            "optimizer" : opt,
            "lr_scheduler" : lr_scheduler
            }


    def initialize(self, io_only=False):

        self._initialize_io(color=self._rank)

        if io_only:
            return

        if self.is_training():
            self.build_lr_schedule()

        with self.default_device_context():
            self.init_network()


            self._net = self._net.to(self.default_device())


            self.print_network_info()

            if self.is_training():
                self.init_optimizer()

            self.init_saver()

            self._global_step = 0

            self.restore_model()

            # If using half precision on the model, convert it now:
            if self.args.run.precision == Precision.bfloat16:
                self._net = self._net.bfloat16()


            if self.is_training():
                if self.args.network.classification.active:
                    with self.default_device_context():
                        weight = torch.tensor([0.16, 0.1666, 0.16666, 0.5]).to(self.default_device())
                        self.loss_calculator = LossCalculator.LossCalculator(self.args, weight=weight)
                else:
                    self.loss_calculator = LossCalculator.LossCalculator(self.args)
            self.acc_calc = AccuracyCalculator.AccuracyCalculator(self.args)

            # For half precision, we disable gradient accumulation.  This is to allow
            # dynamic loss scaling
            if self.args.run.precision == Precision.mixed:
                if self.is_training() and  self.args.mode.optimizer.gradient_accumulation > 1:
                    raise Exception("Can not accumulate gradients in half precision.")

            # self.trace_module()

            if self.args.mode.name == ModeKind.inference:
                self.inference_metrics = {}
                self.inference_metrics['n'] = 0




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

    def restore_model(self):

        state = self.load_state_from_file()

        if state is not None:
            self.restore_state(state)


    def init_optimizer(self):

        from src.config import OptimizerKind

        # get the initial learning_rate:
        initial_learning_rate = self.lr_calculator(self._global_step)


        # IMPORTANT: the scheduler in torch is a multiplicative factor,
        # but I've written it as learning rate itself.  So set the LR to 1.0
        if self.args.mode.optimizer.name == OptimizerKind.rmsprop:
            self._opt = torch.optim.RMSprop(self._net.parameters(), 1.0, eps=1e-6)
        elif self.args.mode.optimizer.name == OptimizerKind.adam:
            self._opt = torch.optim.Adam(self._net.parameters(), 1.0, eps=1e-6, betas=(0.8,0.9))
        elif self.args.mode.optimizer.name == OptimizerKind.adagrad:
            self._opt = torch.optim.Adagrad(self._net.parameters(), 1.0)
        elif self.args.mode.optimizer.name == OptimizerKind.adadelta:
            self._opt = torch.optim.Adadelta(self._net.parameters(), 1.0, eps=1e-6)
        else:
            self._opt = torch.optim.SGD(self._net.parameters(), 1.0)

        # For a regression in pytowrch 1.12.0:
        self._opt.param_groups[0]["capturable"] = False


        if self.args.run.precision == Precision.mixed and self.args.run.compute_mode == ComputeMode.GPU:
            self.scaler = torch.cuda.amp.GradScaler()

    def store_parameters(self, metrics):
        ''' Store all the hyperparameters with MLFLow'''
        flattened_dict = self.flatten(self.args)
        hparams_metrics = {}
        if self.args.mode.name == ModeKind.inference:
            return
        for key in self._hparams_keys:
            if key not in metrics: continue
            hparams_metrics[key] = float(metrics[key].float().cpu())
        if hasattr(self, "_aux_saver") and self._aux_saver is not None:
            self._aux_saver.add_hparams(flattened_dict, hparams_metrics, run_name="hparams")
            self._aux_saver.flush()
        return


    def _calculate_accuracy(self, network_dict, labels_dict):
        ''' Calculate the accuracy.

            Images received here are not sparse but dense.
            This is to ensure equivalent metrics are computed for sparse and dense networks.

        '''

        # Predict the vertex, if needed:
        if self.args.network.vertex.active:
            network_dict['predicted_vertex'] = self.predict_vertex(network_dict)

        return self.acc_calc(network_dict, labels_dict)

    def predict_vertex(self, network_dict):

        # We also flatten to make the argmax operation easier:
        detection_logits = [ n[:,0,:,:].reshape((n.shape[0], -1)) for n in  network_dict['vertex'] ]

        # Extract the index, which comes out flattened:
        predicted_vertex_index = [ torch.argmax(n, dim=1) for n in detection_logits ]


        # Convert flat index to 2D coordinates:
        height_index = [torch.div(p, self.vertex_output_space[1], rounding_mode='floor')  for p in predicted_vertex_index]
        width_index  = [p % self.vertex_output_space[1]  for p in predicted_vertex_index]

        # Extract the regression parameters for every box:
        internal_offsets_height = [ n[:,1,:,:].reshape((n.shape[0], -1)) for n in  network_dict['vertex'] ]
        internal_offsets_width  = [ n[:,2,:,:].reshape((n.shape[0], -1)) for n in  network_dict['vertex'] ]

        # Get the specific regression parameters
        # Creates flat index into the vectors
        batch_size = network_dict['vertex'][0].shape[0]
        batch_indexes = torch.arange(batch_size)

        # Extract the specific internal offsets:
        internal_offsets_height = [ i[batch_indexes, p] for i, p in zip(internal_offsets_height, predicted_vertex_index) ]
        internal_offsets_width  = [ i[batch_indexes, p] for i, p in zip(internal_offsets_width,  predicted_vertex_index) ]

        # Calculate the predicted height as origin + (p+r)*box_size
        predicted_height = [ self.origin[i,0] + (p+r)*self.anchor_size[i,0] for  \
            i, (p,r) in enumerate(zip(height_index, internal_offsets_height)) ]
        predicted_width  = [ self.origin[i,1] + (p+r)*self.anchor_size[i,1] for \
            i, (p,r) in enumerate(zip(width_index, internal_offsets_width)) ]

        # Stack it all together properly:
        vertex_prediction = torch.stack([
            torch.stack(predicted_height, dim=-1),
            torch.stack(predicted_width, dim=-1)
        ], dim=-1)

        return vertex_prediction


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

    def log(self, metrics, saver=''):

        if self._global_step % self.args.mode.logging_iteration == 0:

            self._current_log_time = datetime.datetime.now()

            # Build up a string for logging:
            if self._log_keys != []:
                s = ", ".join(["{0}: {1:.3}".format(key, metrics[key]) for key in self._log_keys])
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
            logger.info("{} Step {} metrics: {}".format(saver, self._global_step, s))

    def summary(self, metrics, saver=""):

        if self._global_step % self.args.mode.summary_iteration == 0:
            for metric in metrics:
                name = metric
                value = metrics[metric]
                if isinstance(value, torch.Tensor):
                    # Cast metrics to 32 bit float
                    value = value.float()

                if saver == "test":
                    self._aux_saver.add_scalar(metric, value, self._global_step)
                else:
                    self._saver.add_scalar(metric, value, self._global_step)


            # try to get the learning rate
            if self.is_training():
                current_lr = self._opt.state_dict()['param_groups'][0]['lr']
                self._saver.add_scalar("learning_rate", current_lr, self._global_step)
            return


    def summary_images(self, logits_image, labels_image, saver=""):

        # if self._global_step % 1 * self.args.mode.summary_iteration == 0:
        if self._global_step % 25 * self.args.mode.summary_iteration == 0 and not self.args.mode.no_summary_images:

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

def create_lightning_module(args, datasets, lr_scheduler=None):

    # Going to build up the lightning module here.

    # Take the first dataset:
    example_ds = next(iter(datasets.values()))

    vertex_meta = create_vertex_meta(args, example_ds.image_meta, example_ds.image_size())


    # Next, create the network:
    network = build_network(args, example_ds.image_size())

    if args.network.classification.active:
        weight = torch.tensor([0.16, 0.1666, 0.16666, 0.5])
        loss_calc = LossCalculator.LossCalculator(args, weight=weight)
    else:
        loss_calc = LossCalculator.LossCalculator(args)
    acc_calc = AccuracyCalculator.AccuracyCalculator(args)


    model = lightning_trainer(args, network, loss_calc, acc_calc, lr_scheduler, vertex_meta)
    return model

def train(args, lightning_model, datasets):

    trainer = pl.Trainer()
    trainer.fit(lightning_model, train_dataloaders=datasets["train"])
    print("Training")