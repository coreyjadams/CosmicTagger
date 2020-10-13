import os
import sys
import time
import tempfile
from collections import OrderedDict

import numpy

import torch
torch.manual_seed(0)

# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# from torch.jit import trace

from src.utils.core.trainercore import trainercore
from src.networks.torch         import LossCalculator



import datetime

# This uses tensorboardX to save summaries and metrics to tensorboard compatible files.

import tensorboardX

class torch_trainer(trainercore):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self,args):
        trainercore.__init__(self, args)
        self._rank = None


    def init_network(self):

        if self.args.conv_mode == "2D" and not self.args.sparse:
            from src.networks.torch.uresnet2D import UResNet
            self._net = UResNet(self.args)

        else:
            if self.args.sparse and self.args.mode != "iotest":
                from src.networks.torch.sparseuresnet3D import UResNet3D
            else:
                from src.networks.torch.uresnet3D       import UResNet3D

            self._net = UResNet3D(self.args, self.larcv_fetcher.image_size())




        if self.args.training:
            self._net.train(True)

        # Here we set up weights using the aggregate metrics for the dataset:

        self._log_keys = ['loss', 'Average/Non_Bkg_Accuracy', 'Average/mIoU']

    def initialize(self, io_only=False):

        self._initialize_io(color=self._rank)

        if io_only:
            return

        if self.args.training:
            self.build_lr_schedule()

        self.init_network()

        self.print_network_info()

        if self.args.mode == "train":
            self.init_optimizer()

        self.init_saver()

        self._global_step = 0

        self.restore_model()

        # If using half precision on the model, convert it now:
        if self.args.precision == "bfloat16":
            self._net = self._net.bfloat16()

        if self.args.compute_mode == "CPU":
            pass
        if self.args.compute_mode == "GPU":
            self._net.cuda()

        self.loss_calculator = LossCalculator.LossCalculator(self.args.loss_balance_scheme)


        # For half precision, we disable gradient accumulation.  This is to allow
        # dynamic loss scaling
        if self.args.precision == "mixed":
            if self.args.gradient_accumulation > 1:
                raise Exception("Can not accumulate gradients in half precision.")

    def print_network_info(self, verbose=False):
        if verbose:
            for name, var in self._net.named_parameters():
                print(name, var.shape)

        self.print("Total number of trainable parameters in this network: {}".format(self.n_parameters()))


    def n_parameters(self):
        n_trainable_parameters = 0
        for name, var in self._net.named_parameters():
            n_trainable_parameters += numpy.prod(var.shape)
            # print(name, var.shape)
        return n_trainable_parameters

    def restore_model(self):

        state = self.load_state_from_file()

        if state is not None:
            self.restore_state(state)


    def init_optimizer(self):

        # get the initial learning_rate:
        initial_learning_rate = self.lr_calculator(self._global_step)

        # IMPORTANT: the scheduler in torch is a multiplicative factor,
        # but I've written it as learning rate itself.  So set the LR to 1.0
        if "RMS" in self.args.optimizer.upper():
            self._opt = torch.optim.RMSprop(self._net.parameters(), 1.0, eps=1e-4)
        else:
            self._opt = torch.optim.Adam(self._net.parameters(), 1.0)

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self._opt, self.lr_calculator, last_epoch=-1)

        if self.args.precision == "mixed" and self.args.compute_mode == "GPU":
            self.scaler = torch.cuda.amp.GradScaler()



    def init_saver(self):

        # This sets up the summary saver:
        dir = self.args.log_directory
        if "torch" not in dir:
            dir = dir + "/torch/"

        self._saver = tensorboardX.SummaryWriter(dir + "/train/")

        if self.args.aux_file is not None and self.args.training:
            self._aux_saver = tensorboardX.SummaryWriter(dir + "/test/")
        elif self.args.aux_file is not None and not self.args.training:
            self._aux_saver = tensorboardX.SummaryWriter(dir + "/val/")
        else:
            self._aux_saver = None
        # This code is supposed to add the graph definition.
        # It doesn't currently work
        # temp_dims = list(dims['image'])
        # temp_dims[0] = 1
        # dummy_input = torch.randn(size=tuple(temp_dims), requires_grad=True)
        # self._saver.add_graph(self._net, (dummy_input,))

        # Here, either restore the weights of the network or initialize it:


    def load_state_from_file(self):
        ''' This function attempts to restore the model from file
        '''

        _, checkpoint_file_path = self.get_model_filepath()

        if not os.path.isfile(checkpoint_file_path):
            return None
        # Parse the checkpoint file and use that to get the latest file path

        with open(checkpoint_file_path, 'r') as _ckp:
            for line in _ckp.readlines():
                if line.startswith("latest: "):
                    chkp_file = line.replace("latest: ", "").rstrip('\n')
                    chkp_file = os.path.dirname(checkpoint_file_path) + "/" + chkp_file
                    self.print("Restoring weights from ", chkp_file)
                    break
        try:
            state = torch.load(chkp_file)
            return state
        except:
            print("Could not load from checkpoint file")
            return None

    def restore_state(self, state):


        self._net.load_state_dict(state['state_dict'])
        if self.args.mode == "train":
            self._opt.load_state_dict(state['optimizer'])
            self.lr_scheduler.load_state_dict(state['scheduler'])

        self._global_step = state['global_step']

        # If using GPUs, move the model to GPU:
        if self.args.compute_mode == "GPU" and self.args.mode == "train":
            for state in self._opt.state.values():
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
            'optimizer'   : self._opt.state_dict(),
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
        if self.args.checkpoint_directory == None:
            file_path= self.args.log_directory  + "/checkpoints/"
        else:
            file_path= self.args.checkpoint_directory  + "/checkpoints/"


        name = file_path + 'model-{}.ckpt'.format(self._global_step)
        checkpoint_file_path = file_path + "checkpoint"

        return name, checkpoint_file_path






    def _calculate_accuracy(self, logits, labels):
        ''' Calculate the accuracy.

            Images received here are not sparse but dense.
            This is to ensure equivalent metrics are computed for sparse and dense networks.

        '''

        accuracy = {}
        accuracy['Average/Total_Accuracy']   = 0.0
        accuracy['Average/Cosmic_IoU']       = 0.0
        accuracy['Average/Neutrino_IoU']     = 0.0
        accuracy['Average/Non_Bkg_Accuracy'] = 0.0
        accuracy['Average/mIoU']             = 0.0


        for plane in [0,1,2]:
            values, predicted_label = torch.max(logits[plane], dim=1)

            correct = (predicted_label == labels[plane].long()).float()

            # We calculate 4 metrics.
            # First is the mean accuracy over all pixels
            # Second is the intersection over union of all cosmic pixels
            # Third is the intersection over union of all neutrino pixels
            # Fourth is the accuracy of all non-zero pixels

            # This is more stable than the accuracy, since the union is unlikely to be ever 0



            # To compute the IoU, we use torch bytetensors which are similar to numpy masks.
            non_zero_locations       = labels[plane] != 0
            neutrino_label_locations = labels[plane] == self.NEUTRINO_INDEX
            cosmic_label_locations   = labels[plane] == self.COSMIC_INDEX

            neutrino_prediction_locations = predicted_label == self.NEUTRINO_INDEX
            cosmic_prediction_locations   = predicted_label == self.COSMIC_INDEX


            non_zero_accuracy = torch.mean(correct[non_zero_locations])

            neutrino_iou = ( (neutrino_prediction_locations & neutrino_label_locations).sum().float() + 0.01) / ( (neutrino_prediction_locations | neutrino_label_locations).sum().float() + 0.01)
            cosmic_iou = ( (cosmic_prediction_locations & cosmic_label_locations).sum().float() + 0.01) /  ((cosmic_prediction_locations | cosmic_label_locations).sum().float() + 0.01)


            accuracy[f'plane{plane}/Total_Accuracy']   = torch.mean(correct)
            accuracy[f'plane{plane}/Cosmic_IoU']       = cosmic_iou
            accuracy[f'plane{plane}/Neutrino_IoU']     = neutrino_iou
            accuracy[f'plane{plane}/Non_Bkg_Accuracy'] = non_zero_accuracy
            accuracy[f'plane{plane}/mIoU']             = 0.5*(cosmic_iou + neutrino_iou)

            accuracy['Average/Total_Accuracy']   += (0.3333333)*torch.mean(correct)
            accuracy['Average/Cosmic_IoU']       += (0.3333333)*cosmic_iou
            accuracy['Average/Neutrino_IoU']     += (0.3333333)*neutrino_iou
            accuracy['Average/Non_Bkg_Accuracy'] += (0.3333333)*non_zero_accuracy
            accuracy['Average/mIoU']             += (0.3333333)*(0.5)*(cosmic_iou + neutrino_iou)



        return accuracy


    def _compute_metrics(self, logits, labels, loss):

        # Call all of the functions in the metrics dictionary:
        metrics = {}

        metrics['loss']     = loss.data
        accuracy = self._calculate_accuracy(logits, labels)
        metrics.update(accuracy)

        return metrics

    def log(self, metrics, saver=''):

        if self._global_step % self.args.logging_iteration == 0:

            self._current_log_time = datetime.datetime.now()

            # Build up a string for logging:
            if self._log_keys != []:
                s = ", ".join(["{0}: {1:.3}".format(key, metrics[key]) for key in self._log_keys])
            else:
                s = ", ".join(["{0}: {1:.3}".format(key, metrics[key]) for key in metrics])

            time_string = []

            if hasattr(self, "_previous_log_time"):
            # try:
                total_images = self.args.minibatch_size
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
            self.print("{} Step {} metrics: {}".format(saver, self._global_step, s))


    def summary(self, metrics,saver=""):

        if self._global_step % self.args.summary_iteration == 0:
            for metric in metrics:
                name = metric
                if saver == "test":
                    self._aux_saver.add_scalar(metric, metrics[metric], self._global_step)
                else:
                    self._saver.add_scalar(metric, metrics[metric], self._global_step)


            # try to get the learning rate
            if self.args.mode == "train":
                self._saver.add_scalar("learning_rate", self._opt.state_dict()['param_groups'][0]['lr'], self._global_step)
            return


    def summary_images(self, logits_image, labels_image, saver=""):

        # if self._global_step % 1 * self.args.summary_iteration == 0:
        if self._global_step % 25 * self.args.summary_iteration == 0 and not self.args.no_summary_images:

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
                labels[labels == self.COSMIC_INDEX] = 0.5
                labels[labels == self.NEUTRINO_INDEX] = 1.0


                prediction[prediction == self.COSMIC_INDEX] = 0.5
                prediction[prediction == self.NEUTRINO_INDEX] = 1.0


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

        if self._global_step % 1 * self.args.summary_iteration == 0:
        # if self._global_step % 25 * self.args.summary_iteration == 0 and not self.args.no_summary_images:
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


    def on_step_end(self):
        pass


    def to_torch(self, minibatch_data, device=None):

        # Convert the input data to torch tensors
        if self.args.compute_mode == "GPU":
            if device is None:
                device = torch.device('cuda')
        else:
            if device is None:
                device = torch.device('cpu')

        for key in minibatch_data:
            if key == 'entries' or key == 'event_ids':
                continue
            if self.args.sparse:
                # if key == 'weight': continue
                if key == 'image':
                    minibatch_data[key] = (
                            torch.tensor(minibatch_data[key][0]).long(),
                            torch.tensor(minibatch_data[key][1], device=device),
                            minibatch_data[key][2],
                        )
                elif key == 'label':
                    minibatch_data[key] = torch.tensor(minibatch_data[key], device=device)
            else:
                if key == 'weight':
                    if self.args.loss_balance_scheme == "none" or self.args.loss_balance_scheme == "focal":
                        continue
                # minibatch_data[key] = torch.tensor(minibatch_data[key], device=device)
                minibatch_data[key] = torch.tensor(minibatch_data[key], device=device)
        if self.args.synthetic:
            minibatch_data['image'] = minibatch_data['image'].float()
            minibatch_data['label'] = minibatch_data['label']
            if self.args.loss_balance_scheme == "even" or self.args.loss_balance_scheme == "light":
                minibatch_data['weight'] = minibatch_data['weight'].float()

        if self.args.precision == "bfloat16":
            minibatch_data["image"] = minibatch_data["image"].bfloat16()

        # self.reduce_precision(minibatch_data)

        return minibatch_data

    # def reduce_precision(self, minibatch_data):
    #     if self.args.precision == "mixed":
    #         minibatch_data['image'] = minibatch_data['image'].half()


    def forward_pass(self, minibatch_data):

        minibatch_data = self.to_torch(minibatch_data)


        # Run a forward pass of the model on the input image:
        logits_image = self._net(minibatch_data['image'])
        labels_image = minibatch_data['label']


        labels_image = labels_image.long()
        labels_image = torch.chunk(labels_image, chunks=3, dim=1)
        shape =  labels_image[0].shape


        # weight = weight.view([shape[0], shape[-3], shape[-2], shape[-1]])

        # print numpy.unique(labels_image.cpu(), return_counts=True)
        labels_image = [ _label.view([shape[0], shape[-2], shape[-1]]) for _label in labels_image ]

        return logits_image, labels_image

    def train_step(self):

        # For a train step, we fetch data, run a forward and backward pass, and
        # if this is a logging step, we compute some logging metrics.


        global_start_time = datetime.datetime.now()

        self._net.train()

        # Reset the gradient values for this step:
        self._opt.zero_grad()

        metrics = {}
        io_fetch_time = 0.0

        for interior_batch in range(self.args.gradient_accumulation):


            # Fetch the next batch of data with larcv
            io_start_time = datetime.datetime.now()
            minibatch_data = self.larcv_fetcher.fetch_next_batch("train",force_pop = True)
            io_end_time = datetime.datetime.now()
            io_fetch_time += (io_end_time - io_start_time).total_seconds()

            # if mixed precision, and cuda, use autocast:
            if self.args.precision == "mixed" and self.args.compute_mode == "GPU":
                with torch.cuda.amp.autocast():
                    logits_image, labels_image = self.forward_pass(minibatch_data)
            else:
                logits_image, labels_image = self.forward_pass(minibatch_data)


            verbose = False

            if verbose: self.print("Completed Forward pass")
            # Compute the loss based on the logits


            loss = self.loss_calculator(labels_image, logits_image)

            if verbose: self.print("Completed loss")

            # Compute the gradients for the network parameters:
            if self.args.precision == "mixed" and self.args.compute_mode == "GPU":
                self.scaler.scale(loss).backward()
            else:
                loss.backward()



            if verbose: self.print("Completed backward pass")


            # Compute any necessary metrics:
            interior_metrics = self._compute_metrics(logits_image, labels_image, loss)

            for key in interior_metrics:
                if key in metrics:
                    metrics[key] += interior_metrics[key]
                else:
                    metrics[key] = interior_metrics[key]

        # Here, make sure to normalize the interior metrics:
        for key in metrics:
            metrics[key] /= self.args.gradient_accumulation

        # Add the global step / second to the tensorboard log:
        try:
            metrics['global_step_per_sec'] = 1./self._seconds_per_global_step
            metrics['images_per_second'] = self.args.minibatch_size / self._seconds_per_global_step
        except:
            metrics['global_step_per_sec'] = 0.0
            metrics['images_per_second'] = 0.0

        metrics['io_fetch_time'] = io_fetch_time

        if verbose: self.print("Calculated metrics")

        step_start_time = datetime.datetime.now()
        # Apply the parameter update:
        if self.args.precision == "mixed" and self.args.compute_mode == "GPU":
            self.scaler.step(self._opt)
            self.scaler.update()
        else:
            self._opt.step()

        self.lr_scheduler.step()

        if verbose: self.print("Updated Weights")
        global_end_time = datetime.datetime.now()

        metrics['step_time'] = (global_end_time - step_start_time).total_seconds()


        self.log(metrics, saver="train")

        if verbose: self.print("Completed Log")

        self.summary(metrics, saver="train")
        self.summary_images(logits_image, labels_image, saver="train")
        # self.graph_summary()
        if verbose: self.print("Summarized")


        # Compute global step per second:
        self._seconds_per_global_step = (global_end_time - global_start_time).total_seconds()

        # Increment the global step value:
        self.increment_global_step()

        return

    def val_step(self):

        # First, validation only occurs on training:
        if not self.args.training: return

        if self.args.synthetic: return
        # Second, validation can not occur without a validation dataloader.
        if self.args.aux_file is None: return

        # perform a validation step
        # Validation steps can optionally accumulate over several minibatches, to
        # fit onto a gpu or other accelerator
        if self._global_step != 0 and self._global_step % self.args.aux_iteration == 0:

            self._net.eval()
            # Fetch the next batch of data with larcv
            # (Make sure to pull from the validation set)
            io_start_time = datetime.datetime.now()
            minibatch_data = self.larcv_fetcher.fetch_next_batch('aux', force_pop = True)
            io_end_time = datetime.datetime.now()

            logits_image, labels_image = self.forward_pass(minibatch_data)

            # Compute the loss based on the logits
            loss = self.loss_calculator(labels_image, logits_image)

            # Compute any necessary metrics:
            metrics = self._compute_metrics(logits_image, labels_image, loss)


            self.log(metrics, saver="test")
            self.summary(metrics, saver="test")
            self.summary_images(logits_image, labels_image, saver="test")

            return

    def checkpoint(self):

        if self._global_step % self.args.checkpoint_iteration == 0 and self._global_step != 0:
            # Save a checkpoint, but don't do it on the first pass
            self.save_model()

    def ana_step(self):

        # First, validation only occurs on training:
        if self.args.training: return

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

        # Convert the input data to torch tensors
        minibatch_data = self.to_torch(minibatch_data)


        # Run a forward pass of the model on the input image:
        with torch.no_grad():
            logits_image, labels_image = self.forward_pass(minibatch_data)


        # If there is an aux file, for ana that means an output file.
        # Call the larcv interface to write data:
        if self.args.aux_file is not None:

            # For writing output, we get the non-zero locations from the labels.
            # Then, we get the neutrino and cosmic scores for those locations in the logits,
            # After applying a softmax.

            # To do this, we just need to capture the following objects:
            # - the dense shape
            # - the location of all non-zero pixels, flattened
            # - the softmax score for all non-zero pixels, flattened.


            # Compute the softmax over all images in the batch:
            softmax = [ torch.nn.Softmax(dim=1)(l) for l in logits_image]


            for b_index in range(self.args.minibatch_size):

                # We want to make sure we get the locations from the non-zero input pixels:
                images = torch.chunk(minibatch_data['image'][b_index,:,:,:], chunks=3, dim=0)

                # Reshape images here to remove the empty index:
                images = [image.squeeze() for image in images]

                # Locations is a list of a tuple of coordinates for each image
                locations = [torch.where(image != 0) for image in images]

                # Shape is a list of shapes for each image:
                shape = [ image.shape for image in images ]

                # Batch softmax is a list of the softmax tensor on each plane
                batch_softmax = [s[b_index] for s in softmax]

                # To get the neutrino scores, we want to access the softmax at the neutrino index
                # And slice over just the non-zero locations:
                neutrino_scores = [ b[self.NEUTRINO_INDEX][locations[plane]]
                    for plane, b in enumerate(batch_softmax) ]
                cosmic_scores   = [ b[self.COSMIC_INDEX][locations[plane]]
                    for plane, b in enumerate(batch_softmax) ]

                # Lastly, flatten the locations.
                # For the unraveled index, there is a complication that torch stores images
                # with [H,W] and larcv3 stores images with [W, H] by default.
                # To solve this -
                # Reverse the shape:
                shape = [ s[::-1] for s in shape ]
                # Go through the locations in reverse:
                locations = [ numpy.ravel_multi_index(l[::-1], s) for l, s in zip(locations, shape) ]



                # Now, package up the objects to send to the file writer:
                neutrino_data = []
                cosmic_data = []
                for plane in range(3):
                    neutrino_data.append({
                        'values' : neutrino_scores[plane].numpy(),
                        'index'  : locations[plane],
                        'shape'  : shape[plane]
                    })
                    cosmic_data.append({
                        'values' : cosmic_scores[plane].numpy(),
                        'index'  : locations[plane],
                        'shape'  : shape[plane]
                    })


                # Write the data through the writer:
                self.larcv_fetcher.write(cosmic_data,
                    producer = "cosmic_prediction",
                    entry    = minibatch_data['entries'][b_index],
                    event_id = minibatch_data['event_ids'][b_index])

                self.larcv_fetcher.write(neutrino_data,
                    producer = "neutrino_prediction",
                    entry    = minibatch_data['entries'][b_index],
                    event_id = minibatch_data['event_ids'][b_index])

        # If the input data has labels available, compute the metrics:
        if 'label' in minibatch_data:
            # Compute the loss
            loss = self.loss_calculator(labels_image, logits_image)

            # Compute the metrics for this iteration:
            metrics = self._compute_metrics(logits_image, labels_image, loss)


            self.log(metrics, saver="ana")
            # self.summary(metrics, saver="test")
            # self.summary_images(logits_image, labels_image, saver="ana")

        return

    def close_savers(self):
        if self._saver is not None:
            self._saver.close()
        if self._aux_saver is not None:
            self._aux_saver.close()
