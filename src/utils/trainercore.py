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

        raise NotImplementedError("You must implement this function")

    def print_network_info(self):
        raise NotImplementedError("You must implement this function")


    def load_model_from_file(self):
        raise NotImplementedError("You must implement this function")

    def set_compute_parameters(self):
        raise NotImplementedError("You must implement this function")


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
        raise NotImplementedError("You must implement this function")



    def init_saver(self):
        raise NotImplementedError("You must implement this function")



    def restore_model(self):
        ''' This function attempts to restore the model from file
        '''
        raise NotImplementedError("You must implement this function")


    def load_state(self, state):

        raise NotImplementedError("You must implement this function")


    def save_model(self):
        '''Save the model to file
        
        '''
        raise NotImplementedError("You must implement this function")

    def get_model_filepath(self):
        '''Helper function to build the filepath of a model for saving and restoring:
        
        
        '''
        raise NotImplementedError("You must implement this function")




    def _calculate_loss(self, labels, logits, weight):
        ''' Calculate the loss.

        returns a single scalar for the optimizer to use.
        '''
        raise NotImplementedError("You must implement this function")

    def _calculate_accuracy(self, logits, labels):
        ''' Calculate the accuracy.

            Images received here are not sparse but dense.
            This is to ensure equivalent metrics are computed for sparse and dense networks.

        '''
        raise NotImplementedError("You must implement this function")


    def _compute_metrics(self, logits, labels, loss):

        raise NotImplementedError("You must implement this function")
        


    def log(self, metrics, saver=''):

        raise NotImplementedError("You must implement this function")


    def summary(self, metrics,saver=""):
        raise NotImplementedError("You must implement this function")



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
        raise NotImplementedError("You must implement this function")

    def on_step_end(self):
        pass

    def on_epoch_end(self):
        pass


    def forward_pass(self, minibatch_data):
        raise NotImplementedError("You must implement this function")

    def train_step(self):
        raise NotImplementedError("You must implement this function")


        return

    def val_step(self):
        raise NotImplementedError("You must implement this function")


 
    def stop(self):
        # Mostly, this is just turning off the io:
        self._larcv_interface.stop()

    def checkpoint(self):

        if self._global_step % FLAGS.CHECKPOINT_ITERATION == 0 and self._global_step != 0:
            # Save a checkpoint, but don't do it on the first pass
            self.save_model()


    def ana_step(self):

        raise NotImplementedError("You must implement this function")


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
