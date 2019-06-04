import os
import sys
import time
import tempfile
from collections import OrderedDict

import numpy

from larcv import larcv_interface


from . import flags
from . import data_transforms
from ..io import io_templates
FLAGS = flags.FLAGS()

import datetime

import tensorboardX
import torch

class trainercore(object):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self,):
        self._larcv_interface = larcv_interface.larcv_interface()
        self._iteration       = 0
        self._global_step     = -1

        self._cleanup         = []

    def __del__(self):
        for f in self._cleanup:
            os.unlink(f.name)
            
    def _initialize_io(self):


        # This is a dummy placeholder, you must check this yourself:
        max_voxels = 1000

        # Use the templates to generate a configuration string, which we store into a temporary file
        if FLAGS.TRAINING:
            config = io_templates.train_io(
                input_file=FLAGS.FILE, 
                data_producer= FLAGS.IMAGE_PRODUCER,
                label_producer= FLAGS.LABEL_PRODUCER, 
                max_voxels=max_voxels)
        else:
            config = io_templates.ana_io(
                input_file=FLAGS.FILE, 
                data_producer= FLAGS.IMAGE_PRODUCER,
                label_producer= FLAGS.LABEL_PRODUCER, 
                max_voxels=max_voxels)


        # Generate a named temp file:
        main_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        main_file.write(config.generate_config_str())

        main_file.close()
        self._cleanup.append(main_file)

        # Prepare data managers:
        io_config = {
            'filler_name' : config._name,
            'filler_cfg'  : main_file.name,
            'verbosity'   : FLAGS.VERBOSITY,
            'make_copy'   : True
        }

        # By default, fetching data and label as the keywords from the file:
        data_keys = OrderedDict({
            'image': 'data', 
            'label': 'label'
            })



        self._larcv_interface.prepare_manager('primary', io_config, FLAGS.MINIBATCH_SIZE, data_keys)

        # All of the additional tools are in case there is a test set up:
        if FLAGS.AUX_FILE is not None:

            if FLAGS.TRAINING:
                config = io_templates.test_io(input_file=FLAGS.AUX_FILE, max_voxels=max_voxels)

                # Generate a named temp file:
                aux_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
                aux_file.write(config.generate_config_str())

                aux_file.close()
                self._cleanup.append(aux_file)
                io_config = {
                    'filler_name' : config._name,
                    'filler_cfg'  : aux_file.name,
                    'verbosity'   : FLAGS.VERBOSITY,
                    'make_copy'   : True
                }

                data_keys = OrderedDict({
                    'image': 'aux_data', 
                    'label': 'aux_label'
                    })
               


                self._larcv_interface.prepare_manager('aux', io_config, FLAGS.AUX_MINIBATCH_SIZE, data_keys)

            else:
                config = io_templates.ana_io(input_file=FLAGS.FILE, max_voxels=max_voxels)
                self._larcv_interface.prepare_writer(FLAGS.AUX_FILE)

    def init_network(self):

        # By default, pass the shape to the network:
        self._net = FLAGS._net(FLAGS.SHAPE)
        # self._net.half()

        if FLAGS.TRAINING: 
            self._net.train(True)

        self._criterion = torch.nn.CrossEntropyLoss(reduction='none')
        
        self._log_keys = ['loss', 'accuracy']


    def print_network_info(self):

        n_trainable_parameters = 0
        for var in self._net.parameters():
            n_trainable_parameters += numpy.prod(var.shape)
        print("Total number of trainable parameters in this network: {}".format(n_trainable_parameters))


    def load_model_from_file(self):

        state = self.restore_model()

        if state is not None:
            self.load_state(state)

    def set_compute_parameters(self):
        # If using half precision on the model, convert it now:
        if FLAGS.MODEL_HALF_PRECISION:
            self._net.half()

        if FLAGS.COMPUTE_MODE == "CPU":
            pass
        if FLAGS.COMPUTE_MODE == "GPU":
            self._net.cuda()


    def initialize(self, io_only=False):

        FLAGS.dump_config()


        self._initialize_io()



        if io_only:
            return

        self.init_network()

        self.print_network_info()

        self.init_optimizer()

        self.init_saver()

        self._global_step = 0


        self.load_model_from_file()

        self.set_compute_parameters()





    def init_optimizer(self):
        if FLAGS.OPTIMIZER == "adam":
            # Create an optimizer:
            if FLAGS.LEARNING_RATE <= 0:
                self._opt = torch.optim.Adam(self._net.parameters())
            else:
                self._opt = torch.optim.Adam(self._net.parameters(), FLAGS.LEARNING_RATE)
        elif FLAGS.OPTIMIZER == "lars":
            raise NotImplementedError("There is no LARS implementation in pytorch")
        else:            
            # Create an optimizer:
            if FLAGS.LEARNING_RATE <= 0:
                self._opt = torch.optim.SGD(self._net.parameters())
            else:
                self._opt = torch.optim.SGD(self._net.parameters(), FLAGS.LEARNING_RATE)




    def init_saver(self):
        # This sets up the summary saver:
        self._saver = tensorboardX.SummaryWriter(FLAGS.LOG_DIRECTORY)

        if FLAGS.AUX_FILE is not None and FLAGS.TRAINING:
            self._aux_saver = tensorboardX.SummaryWriter(FLAGS.LOG_DIRECTORY + "/test/")
        elif FLAGS.AUX_FILE is not None and not FLAGS.TRAINING:
            self._aux_saver = tensorboardX.SummaryWriter(FLAGS.LOG_DIRECTORY + "/val/")
        else:
            self._aux_saver = None

    def restore_model(self):
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
                    print("Restoring weights from ", chkp_file)
                    break

        state = torch.load(chkp_file)
        return state


    def load_state(self, state):

        self._net.load_state_dict(state['state_dict'])
        self._opt.load_state_dict(state['optimizer'])
        self._global_step = state['global_step']

        # If using GPUs, move the model to GPU:
        if FLAGS.COMPUTE_MODE == "GPU":
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
        if FLAGS.CHECKPOINT_DIRECTORY == None:
            file_path= FLAGS.LOG_DIRECTORY  + "/checkpoints/"
        else:
            file_path= FLAGS.CHECKPOINT_DIRECTORY  + "/checkpoints/"


        name = file_path + 'model-{}.ckpt'.format(self._global_step)
        checkpoint_file_path = file_path + "checkpoint"

        return name, checkpoint_file_path




    def _calculate_loss(self, labels, logits):
        ''' Calculate the loss.

        returns a single scalar for the optimizer to use.
        '''
        raise NotImplementedError("You must implement this function")

    def _calculate_accuracy(self, logits, labels):
        ''' Calculate the accuracy.

        '''
        raise NotImplementedError("You must implement this function")


    def _compute_metrics(self, logits, labels, loss):

        # Call all of the functions in the metrics dictionary:
        metrics = {}

        metrics['loss']     = loss.data / FLAGS.LOSS_SCALE 
        accuracy = self._calculate_accuracy(logits, labels)
        metrics.update(accuracy)

        return metrics


    def log(self, metrics, saver=''):


        if self._global_step % FLAGS.LOGGING_ITERATION == 0:
            
            self._current_log_time = datetime.datetime.now()

            # Build up a string for logging:
            if self._log_keys != []:
                s = ", ".join(["{0}: {1:.3}".format(key, metrics[key]) for key in self._log_keys])
            else:
                s = ", ".join(["{0}: {1:.3}".format(key, metrics[key]) for key in metrics])
      

            try:
                s += " ({:.2}s / {:.2} IOs / {:.2})".format(
                    (self._current_log_time - self._previous_log_time).total_seconds(), 
                    metrics['io_fetch_time'],
                    metrics['step_time'])
            except:
                pass

            self._previous_log_time = self._current_log_time

            print("{} Step {} metrics: {}".format(saver, self._global_step, s))


    def summary(self, metrics,saver=""):

        if self._global_step % FLAGS.SUMMARY_ITERATION == 0:
            for metric in metrics:
                name = metric
                if saver == "test":
                    self._aux_saver.add_scalar(metric, metrics[metric], self._global_step)
                else:
                    self._saver.add_scalar(metric, metrics[metric], self._global_step)


            # try to get the learning rate
            self._saver.add_scalar("learning_rate", self._opt.state_dict()['param_groups'][0]['lr'], self._global_step)
            return



    def summary_images(self, logits, labels, saver=""):
        '''Store images to tensorboardX
        
        Create images from the output logits and labels and snapshot them
        Only useful in segmentation networks of some kind.

        Certainly gives garbage in standard classification networks
        
        '''

        if self._global_step % 25 * FLAGS.SUMMARY_ITERATION == 0:

            logits_by_plane = torch.chunk(logits[0], chunks=FLAGS.NPLANES,dim=1)
            labels_by_plane = torch.chunk(labels[0], chunks=FLAGS.NPLANES,dim=0)

            for plane in range(FLAGS.NPLANES):
                val, prediction = torch.max(logits_by_plane[plane], dim=0)
                # This is a reshape and H/W swap:
                prediction = prediction.view(
                    [1, prediction.shape[-2], prediction.shape[-1]]
                    ).permute(0, 2, 1).float()



                labels = labels_by_plane[plane].view(
                    [1, labels_by_plane[plane].shape[-2], labels_by_plane[plane].shape[-1]]
                    ).permute(0, 2, 1)
                # The images are in the format (Plane, W, H)
                # Need to transpose the last two dims in order to meet the (CHW) ordering
                # of tensorboardX


                #  re-write this if you have a number of classes different than 3:
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



    def fetch_next_batch(self, mode='primary', metadata=False):

        minibatch_data = self._larcv_interface.fetch_minibatch_data(mode, fetch_meta_data=metadata)
        minibatch_dims = self._larcv_interface.fetch_minibatch_dims(mode)


        for key in minibatch_data:
            if key == 'entries' or key == 'event_ids':
                continue
            minibatch_data[key] = numpy.reshape(minibatch_data[key], minibatch_dims[key])

        minibatch_data['image']  = data_transforms.larcvsparse_to_dense_2d(minibatch_data['image'], dense_shape=FLAGS.SHAPE)
        minibatch_data['label']  = data_transforms.larcvsparse_to_dense_2d(minibatch_data['label'], dense_shape=FLAGS.SHAPE)



        return minibatch_data


    def increment_global_step(self):
        previous_epoch = int((self._global_step * FLAGS.MINIBATCH_SIZE) / self._epoch_size)
        self._global_step += 1
        current_epoch = int((self._global_step * FLAGS.MINIBATCH_SIZE) / self._epoch_size)

        self.on_step_end()

        if previous_epoch != current_epoch:
            self.on_epoch_end()


    def on_step_end(self):
        pass

    def on_epoch_end(self):
        pass

    def to_torch(self, minibatch_data, device=None):

        # Convert the input data to torch tensors
        if FLAGS.COMPUTE_MODE == "GPU":
            if device is None:
                device = torch.device('cuda')
        else:
            if device is None:
                device = torch.device('cpu')

        for key in minibatch_data:
            if key == 'entries' or key == 'event_ids':
                continue
            if FLAGS.SPARSE:
                minibatch_data[key] = (
                        torch.tensor(minibatch_data[key][0]).long(),
                        torch.tensor(minibatch_data[key][1], device=device),
                        minibatch_data[key][2],
                    )    
            else:
                # minibatch_data[key] = torch.tensor(minibatch_data[key], device=device)
                minibatch_data[key] = torch.tensor(minibatch_data[key], device=device)
                if FLAGS.INPUT_HALF_PRECISION:
                    minibatch_data[key] = minibatch_data[key].half()
        return minibatch_data


    def forward_pass(self, minibatch_data):
        '''Run the model forward with pytorch
        '''

        minibatch_data = self.to_torch(minibatch_data)

        # Run a forward pass of the model on the input image:
        logits = self._net(minibatch_data['image'])
        if 'label' in minibatch_data: 
            labels = minibatch_data['label']
        else:
            labels = None

        return logits, labels

    def train_step(self):

        # For a train step, we fetch data, run a forward and backward pass, and
        # if this is a logging step, we compute some logging metrics.


        global_start_time = datetime.datetime.now()

        self._net.train()
        # Reset the gradient values for this step:
        self._opt.zero_grad()

        # Fetch the next batch of data with larcv
        io_start_time = datetime.datetime.now()
        minibatch_data = self.fetch_next_batch()
        io_end_time = datetime.datetime.now()

        logits, labels = self.forward_pass(minibatch_data)


        verbose = False

        if verbose: print("Completed Forward pass")
        # Compute the loss based on the logits


        loss = self._calculate_loss(labels, logits)
        if verbose: print("Completed loss")

        # Compute the gradients for the network parameters:
        loss.backward()

        # If the loss is scaled, we have to un-scale after the backwards pass
        if FLAGS.LOSS_SCALE != 1.0:
            for param in self._net.parameters():
                param.grad /= FLAGS.LOSS_SCALE

        if verbose: print("Completed backward pass")


        # Compute any necessary metrics:
        metrics = self._compute_metrics(logits, labels, loss)
        


        # Add the global step / second to the tensorboard log:
        try:
            metrics['global_step_per_sec'] = 1./self._seconds_per_global_step
            metrics['images_per_second'] = FLAGS.MINIBATCH_SIZE / self._seconds_per_global_step
        except:
            metrics['global_step_per_sec'] = 0.0
            metrics['images_per_second'] = 0.0

        metrics['io_fetch_time'] = (io_end_time - io_start_time).total_seconds()

        if verbose: print("Calculated metrics")



        step_start_time = datetime.datetime.now()
        # Apply the parameter update:
        self._opt.step()
        if verbose: print("Updated Weights")
        global_end_time = datetime.datetime.now()

        metrics['step_time'] = (global_end_time - step_start_time).total_seconds()


        self.log(metrics, saver="train") 

        if verbose: print("Completed Log")

        self.summary(metrics, saver="train")       
        self.summary_images(logits, labels, saver="train")
        if verbose: print("Summarized")


        # Compute global step per second:
        self._seconds_per_global_step = (global_end_time - global_start_time).total_seconds()

        # Increment the global step value:
        self.increment_global_step()

        # Lastly, call next on the IO:
        self._larcv_interface.next('primary')

        return

    def val_step(self):

        # First, validation only occurs on training:
        if not FLAGS.TRAINING: return

        # Second, validation can not occur without a validation dataloader.
        if FLAGS.AUX_FILE is None: return

        # perform a validation step
        # Validation steps can optionally accumulate over several minibatches, to
        # fit onto a gpu or other accelerator
        if self._global_step % FLAGS.AUX_ITERATION == 0:

            self._net.eval()
            # Fetch the next batch of data with larcv
            # (Make sure to pull from the validation set)
            io_start_time = datetime.datetime.now()
            minibatch_data = self.fetch_next_batch('aux')        
            io_end_time = datetime.datetime.now()

            logits, labels = self.forward_pass(minibatch_data)

            # Compute the loss based on the logits
            loss = self._calculate_loss(labels, logits,)


            # Compute any necessary metrics:
            metrics = self._compute_metrics(logits, labels, loss)
        

            self.log(metrics, saver="test")
            self.summary(metrics, saver="test")
            self.summary_images(logits, labels, saver="test")

            self._larcv_interface.next('aux').next()

            return


 
 
    def stop(self):
        # Mostly, this is just turning off the io:
        self._larcv_interface.stop()

    def checkpoint(self):

        if self._global_step % FLAGS.CHECKPOINT_ITERATION == 0 and self._global_step != 0:
            # Save a checkpoint, but don't do it on the first pass
            self.save_model()


    def ana_step(self):

        # First, validation only occurs on training:
        if FLAGS.TRAINING: return

        # perform a validation step

        # Set network to eval mode
        self._net.eval()
        # self._net.train()

        # Fetch the next batch of data with larcv
        minibatch_data = self.fetch_next_batch(metadata=True)

        # Convert the input data to torch tensors
        minibatch_data = self.to_torch(minibatch_data)
        

        # Run a forward pass of the model on the input image:
        with torch.no_grad():
            logits, labels = self.forward_pass(minibatch_data)
            
        # If the input data has labels available, compute the metrics:
        if labels is not None:
            # Compute the loss
            loss = self._calculate_loss(labels, logits)

            # Compute the metrics for this iteration:
            print("computing metrics for entry ", minibatch_data['entries'][0])
            metrics = self._compute_metrics(logits, labels, loss)


            self.log(metrics, saver="ana")

        self._larcv_interface.next('aux').next()

    def batch_process(self):

        # At the begining of batch process, figure out the epoch size:
        self._epoch_size = self._larcv_interface.size('primary')

        # This is the 'master' function, so it controls a lot


        # Run iterations
        for i in range(FLAGS.ITERATIONS):
            if FLAGS.TRAINING and self._iteration >= FLAGS.ITERATIONS:
                print('Finished training (iteration %d)' % self._iteration)
                self.checkpoint()
                break


            if FLAGS.TRAINING:
                self.val_step()
                self.train_step()
                self.checkpoint()
            else:
                self.ana_step()


        if self._saver is not None:
            self._saver.close()
        if self._aux_saver is not None:
            self._aux_saver.close()
