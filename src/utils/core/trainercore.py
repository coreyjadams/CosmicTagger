import os
import sys
import time
import tempfile
import copy

from collections import OrderedDict

import numpy

from .      import flags
from .      import data_transforms
from ...io  import io_templates

FLAGS = flags.FLAGS()


if not FLAGS.SYNTHETIC:
    from larcv import queueloader

import datetime

class trainercore(object):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self,):
        if not FLAGS.SYNTHETIC:
            if FLAGS.MODE == 'inference':
                mode = 'serial_access'
            else:
                mode = 'random_blocks'
            self._larcv_interface = queueloader.queue_interface(random_access_mode=mode)
        else:
            self.synthetic_images = None
            self.synthetic_labels = None

        self._iteration       = 0
        self._global_step     = -1
        self._val_writer      = None

        self._cleanup         = []

    def cleanup(self):
        for f in self._cleanup:
            try:
                os.unlink(f.name)
            except AttributeError:
                pass

    def _initialize_io(self, color=None):


        if not FLAGS.SYNTHETIC:

            # This is a dummy placeholder, you must check this yourself:
            if 640 in FLAGS.SHAPE:
                max_voxels = 35000
            else:
                max_voxels = 70000

            # Use the templates to generate a configuration string, which we store into a temporary file
            if FLAGS.TRAINING:
                config = io_templates.train_io(
                    input_file      = FLAGS.FILE,
                    data_producer   = FLAGS.IMAGE_PRODUCER,
                    label_producer  = FLAGS.LABEL_PRODUCER,
                    max_voxels      = max_voxels)
            else:
                config = io_templates.ana_io(
                    input_file      = FLAGS.FILE,
                    data_producer   = FLAGS.IMAGE_PRODUCER,
                    label_producer  = FLAGS.LABEL_PRODUCER,
                    max_voxels      =max_voxels)


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

            data_keys = OrderedDict({
                'image': 'data',
                'label': 'label'
                })


            self._larcv_interface.prepare_manager('primary', io_config, FLAGS.MINIBATCH_SIZE, data_keys, color)

            self._larcv_interface.prepare_next('primary')

            # All of the additional tools are in case there is a test set up:
            if FLAGS.AUX_FILE is not None:

                if FLAGS.TRAINING:
                    config = io_templates.test_io(
                        input_file=FLAGS.AUX_FILE,
                        data_producer= FLAGS.IMAGE_PRODUCER,
                        label_producer= FLAGS.LABEL_PRODUCER,
                        max_voxels=max_voxels)

                    # Generate a named temp file:
                    aux_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
                    aux_file.write(config.generate_config_str())

                    aux_file.close()
                    self._cleanup.append(aux_file)
                    io_config = {
                        'filler_name' : config._name,
                        'filler_cfg'  : aux_file.name,
                        'verbosity'   : FLAGS.VERBOSITY,
                        'make_copy'   : False
                    }

                    data_keys = OrderedDict({
                        'image': 'aux_data',
                        'label': 'aux_label'
                        })



                    self._larcv_interface.prepare_manager('aux', io_config, FLAGS.AUX_MINIBATCH_SIZE, data_keys, color)
                    self._larcv_interface.prepare_next('aux')

                else:
                    config = io_templates.output_io(input_file=FLAGS.FILE)
                    print(config.generate_config_str())
                    # Generate a named temp file:
                    out_file_config = tempfile.NamedTemporaryFile(mode='w', delete=False)
                    out_file_config.write(config.generate_config_str())

                    out_file_config.close()
                    self._cleanup.append(out_file_config)
                    self._larcv_interface.prepare_writer(out_file_config.name, FLAGS.AUX_FILE)

            io_dims = self._larcv_interface.fetch_minibatch_dims('primary')

            self.cleanup()



        # Make sure all required dimensions are present:
        else:
            io_dims = {}
            if FLAGS.DATA_FORMAT == "channels_first":
                io_dims['image'] = numpy.asarray(
                    [FLAGS.MINIBATCH_SIZE, 3, FLAGS.SHAPE[0], FLAGS.SHAPE[1]])
                io_dims['label'] = numpy.asarray(
                    [FLAGS.MINIBATCH_SIZE, 3, FLAGS.SHAPE[0], FLAGS.SHAPE[1]])
            else:
                io_dims['image'] = numpy.asarray(
                    [FLAGS.MINIBATCH_SIZE, FLAGS.SHAPE[0], FLAGS.SHAPE[1], 3])
                io_dims['label'] = numpy.asarray(
                    [FLAGS.MINIBATCH_SIZE, FLAGS.SHAPE[0], FLAGS.SHAPE[1], 3])


        if FLAGS.DATA_FORMAT == "channels_last":
            self._channels_dim = -1
        else:
            self._channels_dim = 1

        self._dims = {}
        # Using the sparse IO techniques, we have to manually set the dimensions for the input.

        # Fortunately, everything we need is in the FLAGS object and io object:

        local_minibatch_size = io_dims['image'][0]


        if FLAGS.DATA_FORMAT == "channels_first":
            shape = [local_minibatch_size,] + [3,] + FLAGS.SHAPE
        else:
            shape = [local_minibatch_size,] + FLAGS.SHAPE + [3,]

        self._dims['image'] = numpy.asarray(shape)
        self._dims['label'] = numpy.asarray(shape)


    def init_network(self):
        pass

    def print_network_info(self):
        pass

    def set_compute_parameters(self):
        pass


        self._criterion = torch.nn.CrossEntropyLoss(weight=weight)

        self._log_keys = ['loss', 'accuracy', 'acc-cosmic-iou', 'acc-neutrino-iou']

    def log(self, metrics, kind, step):

        log_string = ""

        log_string += "{} Global Step {}: ".format(kind, step)


        for key in metrics:
            if key in self._log_keys and key != "global_step":
                log_string += "{}: {:.3}, ".format(key, metrics[key])

        if kind == "Train":
            log_string += "Img/s: {:.2} ".format(metrics["images_per_second"])
            log_string += "IO: {:.2} ".format(metrics["io_fetch_time"])
        else:
            log_string.rstrip(", ")

        print(log_string)

        return


    def fetch_next_batch(self, mode='primary', metadata=False, force_pop=False):

        if not FLAGS.SYNTHETIC:
            metadata=True


            pop = True
            if self._iteration == 0 and not force_pop:
                pop = False


            # This brings up the current data
            minibatch_data = self._larcv_interface.fetch_minibatch_data(mode, pop=pop,fetch_meta_data=metadata)
            minibatch_dims = self._larcv_interface.fetch_minibatch_dims(mode)


            for key in minibatch_data:
                if key == 'entries' or key == 'event_ids':
                    continue
                minibatch_data[key] = numpy.reshape(minibatch_data[key], minibatch_dims[key])

            if FLAGS.BALANCE_LOSS and FLAGS.FRAMEWORK == "tensorflow":
                minibatch_data['weight'] = self.compute_weights(minibatch_data['label'])


            if not FLAGS.SPARSE:
                minibatch_data['image']  = data_transforms.larcvsparse_to_dense_2d(minibatch_data['image'], dense_shape=FLAGS.SHAPE)
                minibatch_data['label']  = data_transforms.larcvsparse_to_dense_2d(minibatch_data['label'], dense_shape=FLAGS.SHAPE)
            else:
                minibatch_data['image']  = data_transforms.larcvsparse_to_scnsparse_2d(minibatch_data['image'])
                minibatch_data['label']  = data_transforms.larcvsparse_to_dense_2d(minibatch_data['label'], dense_shape=FLAGS.SHAPE)
            # This preparse the next batch of data:
            self._larcv_interface.prepare_next(mode)

        else:

            # For synthetic data, we can preload the data:
            if self.synthetic_images is None:
                self.prepare_data(dataset_n_entries=12)

            minibatch_data = {}
            if self.synthetic_index + self._dims['image'][0] > len(self.synthetic_images):
                self.synthetic_index = 0

            lower_index = self.synthetic_index
            upper_index = self.synthetic_index + FLAGS.MINIBATCH_SIZE

            minibatch_data['image']  = self.synthetic_images[lower_index:upper_index]
            minibatch_data['label']  = self.synthetic_labels[lower_index:upper_index]
            minibatch_data['weight'] = self.synthetic_weight[lower_index:upper_index]

            self.synthetic_index += 1
            
        return minibatch_data

    def prepare_data(self, dataset_n_entries):

        self.synthetic_index = 0

        shape = copy.copy(self._dims['image'])
        shape[0] = dataset_n_entries

        self.synthetic_images = numpy.random.random_sample(shape)
        self.synthetic_weight = numpy.random.random_sample(shape)
        self.synthetic_labels = numpy.random.randint(low=0, high=3, size=shape)



    def compute_weights(self, labels, boost_labels = None, weight_mode = 2):
        '''
        This is NOT a tensorflow implementation, but a numpy implementation.
        Running on CPUs this might not make a difference.  Running on GPUs
        it might be good to move this to a GPU, but I suspect it's not needed.
        '''
        # Take the labels, and compute the per-label weight

        # Compute weights works on the sparse images, not the dense images.
        # The null-weight is computed on the image shape for dense networks,
        # or based on occupancy on the sparse network.

        # It's done per-batch rather than per image, so:


        # weight modes:
        # 1 - No loss balancing, just run normally
        # 2 - Constant loss balancing: each class is given a weight, but the whole event is normalized
        # 3 - Dynamic loss balancing:  each class is weighted such that it contributes 1/3 of the total weight.
        # 4 - THere is no 4

        x_coords = labels[:,:,:,1]
        y_coords = labels[:,:,:,0]
        val_coords = labels[:,:,:,2]


        # Find the non_zero indexes of the input:
        batch_index, plane_index, voxel_index = numpy.where(val_coords != -999)

        values  = val_coords[batch_index, plane_index, voxel_index]
        x_index = numpy.int32(x_coords[batch_index, plane_index, voxel_index])
        y_index = numpy.int32(y_coords[batch_index, plane_index, voxel_index])

        # Count the types of each label:
        label_values, counts = numpy.unique(values, return_counts=True)

        # Batch size
        batch_size = labels.shape[0]

        # Make sure that if the number of counts is 0 for neutrinos, we fix that
        if len(counts) < 3:
            counts = numpy.insert(counts, 1, 0.1)

        # This computes the *real* number
        # Multiply by 3 planes:
        n_pixels = batch_size * 3* numpy.prod(FLAGS.SHAPE)
        # Correct the empty pixel values in the count:
        counts[0] = n_pixels - counts[1] - counts[2]

        if weight_mode == 1:

            # Now we have the weight values, return it in the proper shape:
            # Prepare output weights:
            class_weights = 0.3333/(counts + 1)

            bkg_weight = class_weights[0]

            weights = numpy.full(values.shape, bkg_weight)
            weights[values==2] = class_weights[2]
            weights[values==1] = class_weights[1]

            if FLAGS.DATA_FORMAT == "channels_first":
                dense_weights = numpy.full([labels.shape[0], 3, FLAGS.SHAPE[0], FLAGS.SHAPE[1]], bkg_weight)
                dense_weights[batch_index,plane_index,y_index,x_index] = weights
            else:
                dense_weights = numpy.full([labels.shape[0], FLAGS.SHAPE[0], FLAGS.SHAPE[1], 3], bkg_weight)
                dense_weights[batch_index,y_index,x_index,plane_index] = weights

        if weight_mode == 2:

            # This mode maintains the weights for everything as if they are unbalanced,
            # however it gives a mild boost to cosmic pixels, and a medium
            # boost to neutrino pixels.

            per_pixel_weight = 1./(numpy.prod(FLAGS.SHAPE))

            bkg_weight = per_pixel_weight
            # Now we have the weight values, return it in the proper shape:
            # Prepare output weights:
            weights = numpy.full(values.shape, bkg_weight)
            weights[values==1] = 10*per_pixel_weight
            weights[values==2] = 1.5*per_pixel_weight





            if FLAGS.DATA_FORMAT == "channels_first":
                dense_weights = numpy.full([labels.shape[0], 3, FLAGS.SHAPE[0], FLAGS.SHAPE[1]], bkg_weight)
                dense_weights[batch_index,plane_index,y_index,x_index] = weights
            else:
                dense_weights = numpy.full([labels.shape[0], FLAGS.SHAPE[0], FLAGS.SHAPE[1], 3], bkg_weight)
                dense_weights[batch_index,y_index,x_index,plane_index] = weights

            # Normalize:
            total_weight = numpy.sum(dense_weights)

            # print("Total_weight: ", total_weight)
            dense_weights *= 1./total_weight




        # # Normalize the weights to sum to 1 for each event:
        return dense_weights

    def on_step_end(self):
        pass

    def on_epoch_end(self):
        pass

    def metrics(self, metrics):
        # This function looks useless, but it is not.
        # It allows a handle to the distributed network to allreduce metrics.
        return metrics

            # For tensorflow, we have to build up an ops list to submit to the
            # session to run.

    def stop(self):
        # Mostly, this is just turning off the io:
        # self._larcv_interface.stop()
        pass


    def batch_process(self, verbose=True):

        # Run iterations
        for self._iteration in range(FLAGS.ITERATIONS):
            if FLAGS.TRAINING and self._iteration >= FLAGS.ITERATIONS:
                print('Finished training (iteration %d)' % self._iteration)
                break

            if FLAGS.MODE == 'train':
                gs = self.train_step()
                self.val_step(gs)
                self.checkpoint(gs)
            elif FLAGS.MODE == 'inference':
                self.ana_step()
            else:
                raise Exception("Don't know what to do with mode ", FLAGS.MODE)

        if FLAGS.MODE == 'inference':
            self._larcv_interface._writer.finalize()
