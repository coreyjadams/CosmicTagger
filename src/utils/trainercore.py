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

        # This function builds the compute graph.
        # Optionally, it can build a 'subset' graph if this mode is

        # Net construction:
        start = time.time()
        sys.stdout.write("Begin constructing network\n")

        # Make sure all required dimensions are present:

        io_dims = self._larcv_interface.fetch_minibatch_dims('primary')

        self._dims = {}
        # Using the sparse IO techniques, we have to manually set the dimensions for the input.
        if "downsample" in FLAGS.FILE:
            if FLAGS.DATA_FORMAT == "channels_last":
                self._dims['image'] = numpy.asarray([io_dims['image'][0],640, 1024, 3])
                self._dims['label'] = numpy.asarray([io_dims['image'][0],640, 1024, 3])
            else:
                self._dims['image'] = numpy.asarray([io_dims['image'][0],3, 640, 1024])
                self._dims['label'] = numpy.asarray([io_dims['image'][0],3, 640, 1024])
        else:
            if FLAGS.DATA_FORMAT == "channels_last":
                self._dims['image'] = numpy.asarray([io_dims['image'][0],1280, 2048, 3])
                self._dims['label'] = numpy.asarray([io_dims['image'][0],1280, 2048, 3])
            else:
                self._dims['image'] = numpy.asarray([io_dims['image'][0],3, 1280, 2048])
                self._dims['label'] = numpy.asarray([io_dims['image'][0],3, 1280, 2048])


        # Call the function to define the inputs
        self._input   = self._initialize_input(self._dims)


        # Add a summary object for the io compute time:
        self._metrics['IO_Fetch_time'] = self._input['io_time']
        # tf.summary.scalar("IO_Fetch_time", self._input['io_time'])

        if FLAGS.MODE == "train":
            # Call the function to define the output
            self._logits  = self._net._build_network(self._input)



            # Here, if the data format is channels_first, we have to reorder the logits tensors
            # To put channels last.  Otherwise it does not work with the softmax tensors.

            if FLAGS.DATA_FORMAT != "channels_last":
                # Split the channel dims apart:
                for i, logit in enumerate(self._logits):
                    n_splits = logit.get_shape().as_list()[1]
                    
                    # Split the tensor apart:
                    split = [tf.squeeze(l, 1) for l in tf.split(logit, n_splits, 1)]
                    
                    # Stack them back together with the right shape:
                    self._logits[i] = tf.stack(split, -1)

            # Apply a softmax and argmax:
            self._outputs = self._create_softmax(self._logits)


            self._accuracy = self._calculate_accuracy(inputs=self._input, outputs=self._outputs)

            # Create the loss function
            self._loss    = self._calculate_loss(inputs=self._input, logits=self._logits)

        end = time.time()
        sys.stdout.write("Done constructing network. ({0:.2}s)\n".format(end-start))

    def print_network_info(self):
        n_trainable_parameters = 0
        for var in tf.trainable_variables():
            n_trainable_parameters += numpy.prod(var.get_shape())
        tf.logging.info("Total number of trainable parameters in this network: {}".format(n_trainable_parameters))


    def load_model_from_file(self):
        raise NotImplementedError("You must implement this function")

    def set_compute_parameters(self):

        self._config = tf.ConfigProto()

        if FLAGS.COMPUTE_MODE == "CPU":
            self._config.inter_op_parallelism_threads = FLAGS.INTER_OP_PARALLELISM_THREADS
            self._config.intra_op_parallelism_threads = FLAGS.INTRA_OP_PARALLELISM_THREADS
        if FLAGS.COMPUTE_MODE == "GPU":
            self._config.gpu_options.allow_growth = True
            # self._config.gpu_options.visible_device_list = str(hvd.local_rank())


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



        if FLAGS.MODE == "train":
            self._sess = tf.train.MonitoredTrainingSession(config=self._config, 
                hooks                 = hooks,
                checkpoint_dir        = FLAGS.LOG_DIRECTORY,
                log_step_count_steps  = FLAGS.LOGGING_ITERATION,
                save_checkpoint_steps = FLAGS.CHECKPOINT_ITERATION)

        elif FLAGS.MODE == "prof":
            self._sess = tf.train.MonitoredTrainingSession(config=self._config, hooks = None,
                checkpoint_dir        = None,
                log_step_count_steps  = None,
                save_checkpoint_steps = None)



    def init_optimizer(self):
        raise NotImplementedError("You must implement this function")



    def init_saver(self):
        raise NotImplementedError("You must implement this function")



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

        return self.restore_from_file(chkp_file)

    def restore_from_file(self. checkpoint_file):
        # Take a checkpoint file and open it and restore it

    def load_state(self, state):

        raise NotImplementedError("You must implement this function")


    def save_model(self):
        '''Save the model to file
        
        '''
        raise NotImplementedError("You must implement this function")

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
