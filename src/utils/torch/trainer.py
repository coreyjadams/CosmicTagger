import os
import sys
import time
import tempfile
from collections import OrderedDict

from src.utils import logging


import numpy
import pandas as pd


import torch
try:
    import intel_extension_for_pytorch as ipex
except:
    pass

# torch.manual_seed(0)

torch.backends.cudnn.benchmark = True


from src.utils.core.trainercore import trainercore
from src.networks.torch         import LossCalculator, AccuracyCalculator
from src.networks.torch         import create_vertex_meta, predict_vertex

import contextlib
@contextlib.contextmanager
def dummycontext():
    yield None

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

import datetime
from src.config import ComputeMode, Precision, ConvMode, ModeKind, DataFormatKind, RunUnit

from . data import create_torch_larcv_dataloader

class torch_trainer(trainercore):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self, args, datasets, lr_schedule, log_keys, hparams_keys):
        trainercore.__init__(self, args)

        # self.datasets = datasets
        self.lr_calculator = lr_schedule
        self.log_keys      = log_keys
        self.hparams_keys  = hparams_keys

        # trainercore.__init__(self, args)
        self.local_df = []

        # Take the first dataset:
        example_ds = next(iter(datasets.values()))

        if self.args.network.vertex.active:
            self.vertex_meta = create_vertex_meta(args, example_ds.image_meta, example_ds.image_size())

        self.latest_metrics = {}


    def init_network(self, image_size, image_meta):
        from src.config import ConvMode

        if self.args.network.conv_mode == ConvMode.conv_2D and not self.args.framework.sparse:
            from src.networks.torch.uresnet2D import UResNet
            self._raw_net = UResNet(self.args.network, image_size)

        else:
            if self.args.framework.sparse and self.args.mode.name != ModeKind.iotest:
                from src.networks.torch.sparseuresnet3D import UResNet3D
            else:
                from src.networks.torch.uresnet3D       import UResNet3D

            self._raw_net = UResNet3D(self.args.network, image_size)

        if self.args.data.data_format == DataFormatKind.channels_last:
            if self.args.run.compute_mode == ComputeMode.XPU:
                self._raw_net = self._raw_net.to("xpu").to(memory_format=torch.channels_last)


        if self.is_training():
             self._raw_net.train(True)

        # Foregoing any fusions as to not disturb the existing ingestion pipeline
        if self.is_training() and self.args.mode.quantization_aware:
            self._raw_net.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            self._net = torch.quantization.prepare_qat(self._raw_net)
        else:
            self._net = self._raw_net



    def initialize(self, datasets):

        example_ds = next(iter(datasets.values()))

        with self.default_device_context():
            self.init_network(example_ds.image_size(), example_ds.image_meta)


            self._net = self._net.to(self.default_device())

            self.print_network_info()

            if self.is_training():
                self.init_optimizer()

            # Initialize savers:
            dir = self.args.output_dir
            if not self.args.run.distributed or self.rank == 0:
                self.savers = {
                    ds_name : SummaryWriter(dir + f"/{ds_name}/")
                    for ds_name in datasets.keys()
                }
            else:
                self.savers = {ds_name : None for ds_name in datasets.keys()}


            self._global_step = 0

            self.restore_model()

            # If using half precision on the model, convert it now:
            if self.args.run.precision == Precision.bfloat16:
                self._net = self._net.bfloat16()


            if self.is_training():
                if self.args.network.classification.active:
                    with self.default_device_context():
                        weight = torch.tensor([0.16, 0.1666, 0.16666, 0.5]).to(self.default_device())
                        self.loss_calculator = LossCalculator(self.args, weight=weight)
                else:
                    self.loss_calculator = LossCalculator(self.args)
            self.acc_calc = AccuracyCalculator(self.args)

            # For half precision, we disable gradient accumulation.  This is to allow
            # dynamic loss scaling
            if self.args.run.precision == Precision.mixed:
                if self.is_training() and  self.args.mode.optimizer.gradient_accumulation > 1:
                    raise Exception("Can not accumulate gradients in half precision.")

            # example_batch = next(iter(example_ds))

            # self.trace_module(example_batch)

            if self.args.mode.name == ModeKind.inference:
                self.inference_metrics = {}
                self.inference_metrics['n'] = 0

        # Turn the datasets into torch dataloaders
        for key in datasets.keys():
            datasets[key] = create_torch_larcv_dataloader(
                datasets[key], self.args.run.minibatch_size,
                device = self.default_device())


    def trace_module(self, example_batch):
        #
        # if self.args.run.precision == Precision.mixed:
        #     logger.warning("Tracing not available with mixed precision, sorry")
        #     return

        # Trace the module:
        # minibatch_data = self.larcv_fetcher.fetch_next_batch("train",force_pop = True)
        # example_inputs = self.to_torch(minibatch_data)
        # Run a forward pass of the model on the input image:

        inputs = torch.tensor(example_batch["image"]).cuda()

        if self.args.run.precision == Precision.mixed and self.args.run.compute_mode == ComputeMode.CUDA:
            with torch.cuda.amp.autocast():
                self._net = torch.jit.trace(self._net, inputs , strict=False)
        else:
            self._net = torch.jit.trace(self._net, inputs , strict=False)




    def print_network_info(self, verbose=False):
        logger = logging.getLogger("CosmicTagger")
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




        # IMPORTANT: the scheduler in torch is a multiplicative factor,
        # but I've written it as learning rate itself.  So set the LR to 1.0
        if self.args.mode.optimizer.name == OptimizerKind.rmsprop:
            self.opt = torch.optim.RMSprop(self._net.parameters(), 1.0, eps=1e-6)
        elif self.args.mode.optimizer.name == OptimizerKind.adam:
            self.opt = torch.optim.Adam(self._net.parameters(), 1.0, eps=1e-6, betas=(0.8,0.9))
        elif self.args.mode.optimizer.name == OptimizerKind.adagrad:
            self.opt = torch.optim.Adagrad(self._net.parameters(), 1.0)
        elif self.args.mode.optimizer.name == OptimizerKind.adadelta:
            self.opt = torch.optim.Adadelta(self._net.parameters(), 1.0, eps=1e-6)
        else:
            self.opt = torch.optim.SGD(self._net.parameters(), 1.0)

        # For a regression in pytowrch 1.12.0:
        self.opt.param_groups[0]["capturable"] = False

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, self.lr_calculator, last_epoch=-1)

        if self.args.run.precision == Precision.mixed and self.args.run.compute_mode == ComputeMode.CUDA:
            self.scaler = torch.cuda.amp.GradScaler()

    def load_state_from_file(self):
        ''' This function attempts to restore the model from file
        '''

        logger = logging.getLogger("CosmicTagger")
        def check_inference_weights_path(file_path):

            # Look for the "checkpoint" file:
            checkpoint_file_path = file_path + "checkpoint"
            # If it exists, open it and read the latest checkpoint:
            if os.path.isfile(checkpoint_file_path):
                return checkpoint_file_path


        # First, check if the weights path is set:
        if self.args.mode.weights_location != "":
            checkpoint_file_path = check_inference_weights_path(self.args.mode.weights_location)
        else:
            _, checkpoint_file_path = self.get_model_filepath()


        if not os.path.isfile(checkpoint_file_path):
            logger.info("No checkpoint file found, restarting from scratch")
            return None
        # Parse the checkpoint file and use that to get the latest file path

        with open(checkpoint_file_path, 'r') as _ckp:
            for line in _ckp.readlines():
                if line.startswith("latest: "):
                    chkp_file = line.replace("latest: ", "").rstrip('\n')
                    chkp_file = os.path.dirname(checkpoint_file_path) + "/" + chkp_file
                    logger.info(f"Restoring weights from {chkp_file}")
                    break
        try:
            state = torch.load(chkp_file)
            return state
        except:
            logger.warning("Could not load from checkpoint file")
            return None

    def restore_state(self, state):

        new_state_dict = {}
        for key in state['state_dict']:
            if key.startswith("module."):
                new_key = key.lstrip("module.")
            else:
                new_key = key
            new_state_dict[new_key] = state['state_dict'][key]

        state['state_dict'] = new_state_dict

        self._net.load_state_dict(state['state_dict'])
        if self.is_training():
            self.opt.load_state_dict(state['optimizer'])
            self.lr_scheduler.load_state_dict(state['scheduler'])

        self._global_step = state['global_step']

        # If using GPUs, move the model to GPU:
        if self.args.run.compute_mode == ComputeMode.CUDA and self.is_training():
            for state in self.opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        return True

    def save_model(self):
        '''Save the model to file

        '''

        current_file_path, checkpoint_file_path = self.get_model_filepath()

        # save the model state into the file path:
        state_dict = {
            'global_step' : self._global_step,
            'state_dict'  : self._net.state_dict(),
            'optimizer'   : self.opt.state_dict(),
            'scheduler'   : self.lr_scheduler.state_dict(),
        }

        # Make sure the path actually exists:
        if not os.path.isdir(os.path.dirname(current_file_path)):
            os.makedirs(os.path.dirname(current_file_path))

        torch.save(state_dict, current_file_path)

        # Parse the checkpoint file to see what the last checkpoints were:

        # Keep only the last 5 checkpoints
        n_keep = 5


        past_checkpoint_files = {}
        try:
            with open(checkpoint_file_path, 'r') as _chkpt:
                for line in _chkpt.readlines():
                    line = line.rstrip('\n')
                    vals = line.split(":")
                    if vals[0] != 'latest':
                        past_checkpoint_files.update({int(vals[0]) : vals[1].replace(' ', '')})
        except:
            pass


        # Remove the oldest checkpoints while the number is greater than n_keep
        while len(past_checkpoint_files) >= n_keep:
            min_index = min(past_checkpoint_files.keys())
            file_to_remove = os.path.dirname(checkpoint_file_path) + "/" + past_checkpoint_files[min_index]
            os.remove(file_to_remove)
            past_checkpoint_files.pop(min_index)



        # Update the checkpoint file
        with open(checkpoint_file_path, 'w') as _chkpt:
            _chkpt.write('latest: {}\n'.format(os.path.basename(current_file_path)))
            _chkpt.write('{}: {}\n'.format(self._global_step, os.path.basename(current_file_path)))
            for key in past_checkpoint_files:
                _chkpt.write('{}: {}\n'.format(key, past_checkpoint_files[key]))


    def get_model_filepath(self):
        '''Helper function to build the filepath of a model for saving and restoring:


        '''

        # Find the base path of the log directory
        file_path= self.args.output_dir  + "/checkpoints/"


        name = file_path + 'model-{}.ckpt'.format(self._global_step)
        checkpoint_file_path = file_path + "checkpoint"

        return name, checkpoint_file_path

    def store_parameters(self, metrics):
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


    def _calculate_accuracy(self, network_dict, labels_dict, batch_reduce=True):
        ''' Calculate the accuracy.

            Images received here are not sparse but dense.
            This is to ensure equivalent metrics are computed for sparse and dense networks.

        '''


        # Predict the vertex, if needed:
        if self.args.network.vertex.active:
            network_dict['predicted_vertex'] = predict_vertex(network_dict, self.vertex_meta)

        return self.acc_calc(network_dict, labels_dict, batch_reduce)


    def _compute_metrics(self, network_dict, labels_dict, loss_dict, batch_reduce=True):

        with torch.no_grad():

            metrics = self._calculate_accuracy(network_dict, labels_dict, batch_reduce)

            if loss_dict is not None:
                for key in loss_dict:
                    metrics[f'loss/{key}'] = loss_dict[key].detach()

            ## TODO - add vertex resolution????

        return metrics


    def summary(self, metrics, saver):

        if self._global_step % self.args.mode.summary_iteration == 0:
            for metric in metrics:
                name = metric
                value = metrics[metric]
                if isinstance(value, torch.Tensor):
                    # Cast metrics to 32 bit float
                    value = value.float()
                saver.add_scalar(metric, value, self._global_step)




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



    def default_device_context(self):


        if self.args.run.compute_mode == ComputeMode.CUDA:
            return torch.cuda.device(0)
        elif self.args.run.compute_mode == ComputeMode.XPU:
            # return contextlib.nullcontext
            try:
                return ipex.xpu.device("xpu:0")
            except:
                pass
            try:
                return ipex.device("xpu:0")
            except:
                pass
            return contextlib.nullcontext()
        else:
            return contextlib.nullcontext()

    def default_device(self):

        if self.args.run.compute_mode == ComputeMode.CUDA:
            return torch.device("cuda")
        elif self.args.run.compute_mode == ComputeMode.XPU:
            device = torch.device("xpu")
        else:
            device = torch.device('cpu')
        return device


    def forward_pass(self, minibatch_data, net=None):


        with self.default_device_context():

            if self.args.run.compute_mode == ComputeMode.XPU:
                if self.args.data.data_format == DataFormatKind.channels_last:
                    # minibatch_data["image"] = minibatch_data["image"].to(memory_format=torch.channels_last)
                    # minibatch_data["label"] = minibatch_data['label'].to(memory_format=torch.channels_last)

                    minibatch_data['label'].to(memory_format=torch.channels_last)
                    minibatch_data["image"].to(memory_format=torch.channels_last)


            labels_dict = {
                "segmentation" : torch.chunk(minibatch_data['label'].long(), chunks=3, dim=1),

            }
            if self.args.network.classification.active or self.args.network.vertex.active:
                labels_dict.update({"event_label"  : minibatch_data['event_label']})
            if self.args.network.vertex.active:
                labels_dict.update({"vertex"  : minibatch_data['vertex']})

            # Run a forward pass of the model on the input image:
            if net is None:
                network_dict = self._net(minibatch_data['image'])
            else:
                network_dict = net(minibatch_data['image'])

            shape =  labels_dict["segmentation"][0].shape


            # print numpy.unique(labels_image.cpu(), return_counts=True)
            labels_dict["segmentation"] = [
                _label.view([shape[0], shape[-2], shape[-1]])
                    for _label in labels_dict["segmentation"]
            ]

        return network_dict, labels_dict

    def train_step(self, minibatch_data):

        # For a train step, we fetch data, run a forward and backward pass, and
        # if this is a logging step, we compute some logging metrics.


        global_start_time = datetime.datetime.now()

        self._net.train()

        # Reset the gradient values for this step:
        self.opt.zero_grad()

        metrics = {}
        io_fetch_time = 0.0

        grad_accum = self.args.mode.optimizer.gradient_accumulation

        use_cuda=torch.cuda.is_available()
        with self.default_device_context():
            for interior_batch in range(grad_accum):

                # Fetch the next batch of data with larcv
                # io_start_time = datetime.datetime.now()
                # # with self.timing_context("io"):
                # #     # // TODO change force_pop to True!
                # #     minibatch_data = self.larcv_fetcher.fetch_next_batch("train",force_pop = True)
                # io_end_time = datetime.datetime.now()
                # io_fetch_time += (io_end_time - io_start_time).total_seconds()



                if self.args.run.profile:
                    if not self.args.run.distributed or self.rank == 0:
                        autograd_prof = torch.autograd.profiler.profile(use_cuda = use_cuda)
                    else:
                        autograd_prof = dummycontext()
                else:
                    autograd_prof = dummycontext()

                with autograd_prof as prof:

                    # if mixed precision, and cuda, use autocast:
                    with self.timing_context("forward"):
                        if self.args.run.precision == Precision.mixed and self.args.run.compute_mode == ComputeMode.CUDA:
                            with torch.cuda.amp.autocast():
                                network_dict, labels_dict = self.forward_pass(minibatch_data)
                        else:
                            network_dict, labels_dict = self.forward_pass(minibatch_data)


                    # Compute the loss based on the network_dict
                    with self.timing_context("loss"):
                        loss, loss_metrics = self.loss_calculator(labels_dict, network_dict)

                    # if loss.isnan(): exit()
                    # Compute the gradients for the network parameters:
                    with self.timing_context("backward"):
                        if self.args.run.precision == Precision.mixed and self.args.run.compute_mode == ComputeMode.CUDA:
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()


                    # Compute any necessary metrics:
                    with self.timing_context("metrics"):
                        interior_metrics = self._compute_metrics(network_dict, labels_dict, loss_metrics)

                        for key in interior_metrics:
                            if key in metrics:
                                metrics[key] += interior_metrics[key]
                            else:
                                metrics[key] = interior_metrics[key]

                # save profile data per step
                if self.args.run.profile:
                    if not self.args.run.distributed or self.rank == 0:
                        prof.export_chrome_trace("timeline_" + str(self._global_step) + ".json")

            step_start_time = datetime.datetime.now()
            # Putting the optimization step here so the metrics and such can be concurrent:
            with self.timing_context("optimizer"):
                # Apply the parameter update:
                if self.args.run.precision == Precision.mixed and self.args.run.compute_mode == ComputeMode.CUDA:
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    self.opt.step()
                self.lr_scheduler.step()
            global_end_time = datetime.datetime.now()

            # Here, make sure to normalize the interior metrics:
            for key in metrics:
                metrics[key] /= grad_accum

            # Add the global step / second to the tensorboard log:
            try:
                metrics['global_step_per_sec'] = 1./self._seconds_per_global_step
                metrics['images_per_second'] = self.args.run.minibatch_size / self._seconds_per_global_step
            except:
                metrics['global_step_per_sec'] = 0.0
                metrics['images_per_second'] = 0.0

            metrics['io_fetch_time'] = io_fetch_time




            metrics['step_time'] = (global_end_time - step_start_time).total_seconds()


            with self.timing_context("log"):
                self.log(metrics, self.log_keys, saver="train")


            with self.timing_context("summary"):

                # try to get the learning rate
                current_lr = self.opt.state_dict()['param_groups'][0]['lr']
                metrics["learning_rate"] = current_lr

                self.summary(metrics, saver=self.savers["train"])
                self.summary_images(network_dict["segmentation"], labels_dict["segmentation"], saver=self.savers["train"])
                # self.graph_summary()


            # Compute global step per second:
            self._seconds_per_global_step = (global_end_time - global_start_time).total_seconds()

            # Increment the global step value:
            self.increment_global_step()

        return

    def val_step(self, minibatch_data, store=True):

        # First, validation only occurs on training:
        if not self.is_training(): return

        if self.args.data.synthetic: return

        # perform a validation step
        # Validation steps can optionally accumulate over several minibatches, to
        # fit onto a gpu or other accelerator
        self._net.eval()
        if self.args.run.compute_mode == ComputeMode.CPU:
            # Quantization not supported on CUDA
            val_net = torch.quantization.convert(self._net)
        else:
            val_net = self._net


        # if mixed precision, and cuda, use autocast:
        if self.args.run.precision == Precision.mixed and self.args.run.compute_mode == ComputeMode.CUDA:
            with torch.cuda.amp.autocast():
                network_dict, labels_dict = self.forward_pass(minibatch_data, net=val_net)
        else:
            network_dict, labels_dict = self.forward_pass(minibatch_data, net=val_net)



        # Compute the loss based on the network_dict
        loss, loss_metrics = self.loss_calculator(labels_dict, network_dict)


        # Compute any necessary metrics:
        metrics = self._compute_metrics(network_dict, labels_dict, loss_metrics)

        if store:
            self.log(metrics, self.log_keys, saver="val")
            self.summary(metrics, saver=self.savers["val"])
            self.summary_images(network_dict["segmentation"], labels_dict["segmentation"], saver=self.savers["val"])


        # Store these for hyperparameter logging.
        self.latest_metrics = metrics
        return metrics

    def finalize_val_metrics(self, metrics_list):
        metrics = {
            key : torch.mean(
                [ m[key] for m in metrics_list]
            ) for key in metrics_list[0].keys()
        }
        self.log(metrics, self.log_keys, saver="val")
        self.summary(metrics, saver=self.savers["val"])
        self.summary_images(network_dict["segmentation"], labels_dict["segmentation"], saver=self.savers["val"])


    def checkpoint(self):

        if self.args.run.run_units == RunUnit.iteration:

            if self._global_step % self.args.mode.checkpoint_iteration == 0 and self._global_step != 0:
                # Save a checkpoint, but don't do it on the first pass
                self.save_model()
        else:
            # self._epoch % self.args.mode.checkpoint_iteration == 0 and
            if self._epoch_end:
                self.save_model()

    def ana_step(self):

        # First, validation only occurs on training:
        if self.is_training(): return

        # perform a validation step

        # Set network to eval mode
        self._net.eval()
        # self._net.train()

        # Fetch the next batch of data with larcv
        if self._iteration == 0:
            force_pop = False
        else:
            force_pop = True
        minibatch_data = self.larcv_fetcher.fetch_next_batch("train", force_pop=force_pop)


        # Run a forward pass of the model on the input image:
        with torch.no_grad():
            if self.args.run.precision == Precision.mixed and self.args.run.compute_mode == ComputeMode.CUDA:
                with torch.cuda.amp.autocast():
                    logits_dict, labels_dict = self.forward_pass(minibatch_data)
            else:
                logits_dict, labels_dict = self.forward_pass(minibatch_data)



        # If the input data has labels available, compute the metrics:
        if 'label' in minibatch_data:
            # Compute the loss
            # loss = self.loss_calculator(labels_dict, logits_dict)

            # Compute the metrics for this iteration:
            metrics = self._compute_metrics(logits_dict, labels_dict, loss_dict=None, batch_reduce=False)

            # We can count the number of neutrino id'd pixels per plane:
            n_neutrino_pixels = [ torch.sum(torch.argmax(p, axis=1) == 2, axis=(1,2)) for p in logits_dict["segmentation"]]
            predicted_vertex = predict_vertex(logits_dict, self.vertex_meta)
            predicted_label = torch.softmax(logits_dict["event_label"],axis=1)
            predicted_label = torch.argmax(predicted_label, axis=1)
            prediction_score = torch.max(predicted_label)
            additional_info = {
                "index"            : numpy.asarray(minibatch_data["entries"]),
                "event_id"         : numpy.asarray(minibatch_data["event_ids"]),
                "energy"           : minibatch_data["vertex"]["energy"],
                # "predicted_vertex" : predicted_vertex,
                "predicted_vertex0h" : predicted_vertex[:,0,0],
                "predicted_vertex0w" : predicted_vertex[:,0,1],
                "predicted_vertex1h" : predicted_vertex[:,1,0],
                "predicted_vertex1w" : predicted_vertex[:,1,1],
                "predicted_vertex2h" : predicted_vertex[:,2,0],
                "predicted_vertex2w" : predicted_vertex[:,2,1],
                # "predicted_vertex2" : predicted_vertex[:,2,:],
                # "true_vertex"      : labels_dict["vertex"]["xy_loc"],
                "true_vertex0h"      : labels_dict["vertex"]["xy_loc"][:,0,0],
                "true_vertex0w"      : labels_dict["vertex"]["xy_loc"][:,0,1],
                "true_vertex1h"      : labels_dict["vertex"]["xy_loc"][:,1,0],
                "true_vertex1w"      : labels_dict["vertex"]["xy_loc"][:,1,1],
                "true_vertex2h"      : labels_dict["vertex"]["xy_loc"][:,2,0],
                "true_vertex2w"      : labels_dict["vertex"]["xy_loc"][:,2,1],
                "vertex_3dx"         : minibatch_data["vertex"]["xyz_loc"]["_x"],
                "vertex_3dy"         : minibatch_data["vertex"]["xyz_loc"]["_y"],
                "vertex_3dz"         : minibatch_data["vertex"]["xyz_loc"]["_z"],
                "N_neut_pixels0"     : n_neutrino_pixels[0],
                "N_neut_pixels1"     : n_neutrino_pixels[1],
                "N_neut_pixels2"     : n_neutrino_pixels[2],
                "predicted_label"  : predicted_label,
                "prediction_score"  : prediction_score,
                "true_label"       : labels_dict["event_label"],
            }

            # Move everything in the dictionary to CPU:
            additional_info.update(metrics)
            for key in additional_info.keys():
                if type(additional_info[key]) == torch.Tensor:
                    additional_info[key] = additional_info[key].cpu().numpy()

            self.local_df.append(pd.DataFrame.from_dict(additional_info))

            # Reduce the metrics over the batch size here:
            metrics = { key : torch.mean(metrics[key], axis=0) for key in metrics.keys() }

            self.accumulate_metrics(metrics)

            # print(minibatch_data)
            self.log(metrics, saver="ana")

        self._global_step += 1

        return

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
        logger = logging.getLogger("CosmicTagger")

        if hasattr(self, "local_df") and self.local_df is not None:
            local_df = pd.concat(self.local_df)
            outdir = self.args.output_dir
            print(outdir)
            local_df.to_csv(f"{outdir}/rank_{self.rank}_{self.args.run.id}.csv")

        if not hasattr(self, "inference_metrics"):
            return
        n = self.inference_metrics["n"]
        total_entries = n*self.args.run.minibatch_size
        logger.info(f"Inference report: {n} batches processed for {total_entries} entries.")
        for key in self.inference_metrics:
            if key == 'n' or '_sq' in key: continue
            value = self.inference_metrics[key] / n
            logger.info(f"  {key}: {value:.4f}")

    def close_savers(self):
        if hasattr(self, "savers"):
            for saver in self.savers.values():
                saver.flush()
                saver.close()

    def exit(self):
        self.store_parameters(self.latest_metrics)
        self.close_savers()
        trainercore.exit(self)

        super()
