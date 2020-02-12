import os
import sys
import time
import tempfile
import copy

from collections import OrderedDict

import numpy

# larcv_fetcher can also do synthetic IO without any larcv installation
from . larcvio import larcv_fetcher

import datetime



class trainercore(object):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    NEUTRINO_INDEX = 2
    COSMIC_INDEX   = 1


    def __init__(self, args):

        self._iteration    = 0
        self._global_step  = -1
        self.args          = args
        self.larcv_fetcher = larcv_fetcher.larcv_fetcher(
            mode        = args.mode, 
            distributed = args.distributed,
            downsample  = args.downsample_images, 
            dataformat  = args.data_format,
            synthetic   = args.synthetic,
            sparse      = args.sparse )

        if args.data_format == "channels_first": self._channels_dim = 1
        if args.data_format == "channels_last" : self._channels_dim = -1



    def _initialize_io(self, color=None):

        self._train_data_size = self.larcv_fetcher.prepare_cosmic_sample(
            "train", self.args.file, self.args.minibatch_size)

        if self.args.aux_file is not None:
            self._aux_data_size = self.larcv_fetcher.prepare_cosmic_sample(
                "aux", self.args.aux_file, self.args.minibatch_size)


        # self._dims['image'] = self.larcv_fetcher.batch_dims(self.args.minibatch_size)
        # self._dims['label'] = self.larcv_fetcher.batch_dims(self.args.minibatch_size)


    def init_network(self):
        pass

    def print_network_info(self):
        pass

    def set_compute_parameters(self):
        pass


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


    # def fetch_next_batch(self, mode='train', metadata=False, force_pop=False):

    #     if not FLAGS.SYNTHETIC:
    #         metadata=True


    #         pop = True
    #         if self._iteration == 0 and not force_pop:
    #             pop = False


    #         # This brings up the current data
    #         self._larcv_interface.prepare_next(mode)
    #         minibatch_data = self._larcv_interface.fetch_minibatch_data(mode, pop=pop,fetch_meta_data=metadata)
    #         minibatch_dims = self._larcv_interface.fetch_minibatch_dims(mode)


    #         for key in minibatch_data:
    #             if key == 'entries' or key == 'event_ids':
    #                 continue
    #             minibatch_data[key] = numpy.reshape(minibatch_data[key], minibatch_dims[key])

    #         # if FLAGS.LOSS_BALANCE_SCHEME != "none":
    #         minibatch_data['weight'] = self.compute_weights(minibatch_data['label'])




    #         if not FLAGS.SPARSE:
    #             minibatch_data['image']  = data_transforms.larcvsparse_to_dense_2d(
    #                 minibatch_data['image'], dense_shape=self.full_image_shape)
    #             minibatch_data['label']  = data_transforms.larcvsparse_to_dense_2d(
    #                 minibatch_data['label'], dense_shape=self.full_image_shape)
    #         else:
    #             minibatch_data['image']  = data_transforms.larcvsparse_to_scnsparse_2d(
    #                 minibatch_data['image'])
    #             minibatch_data['label']  = data_transforms.larcvsparse_to_dense_2d(
    #                 minibatch_data['label'], dense_shape=self.full_image_shape)


    #     else:

    #         # For synthetic data, we can preload the data:
    #         if self.synthetic_images is None:
    #             self.prepare_data(dataset_n_entries=12)

    #         minibatch_data = {}
    #         if self.synthetic_index + self._dims['image'][0] > len(self.synthetic_images):
    #             self.synthetic_index = 0

    #         lower_index = self.synthetic_index
    #         upper_index = self.synthetic_index + FLAGS.MINIBATCH_SIZE

    #         minibatch_data['image']  = self.synthetic_images[lower_index:upper_index]
    #         minibatch_data['label']  = self.synthetic_labels[lower_index:upper_index]
    #         minibatch_data['weight'] = self.compute_weights(minibatch_data['label'])

    #         self.synthetic_index += 1

    #     return minibatch_data


    def compute_weights(self, labels):

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
        # "none" - No loss balancing, just run normally
        # "light" - Constant loss balancing: each class is given a weight, but the whole event is normalized
        # "even" - Dynamic loss balancing:  each class is weighted such that it contributes 1/3 of the total weight.
        # "focal" - Compute the focal loss, which is done in the framework loss functions

        if FLAGS.LOSS_BALANCE_SCHEME == "focal": return None
        if FLAGS.LOSS_BALANCE_SCHEME == "none": return None

        x_coords = labels[:,:,:,0]
        y_coords = labels[:,:,:,1]
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
            counts = numpy.insert(counts, self.NEUTRINO_INDEX, 0.1)

        # This computes the *real* number
        # Multiply by 3 planes:
        n_pixels = batch_size * 3* numpy.prod(self.image_shape)
        # Correct the empty pixel values in the count:
        counts[0] = n_pixels - counts[1] - counts[2]

        if FLAGS.LOSS_BALANCE_SCHEME == "even":

            # Now we have the weight values, return it in the proper shape:
            # Prepare output weights:
            class_weights = 0.3333/(counts + 1)

            bkg_weight = class_weights[0]

            weights = numpy.full(values.shape, bkg_weight)
            weights[values==self.COSMIC_INDEX]   = class_weights[self.COSMIC_INDEX]
            weights[values==self.NEUTRINO_INDEX] = class_weights[self.NEUTRINO_INDEX]

            if FLAGS.DATA_FORMAT == "channels_first":
                dense_weights = numpy.full([labels.shape[0], 3, self.image_shape[0], self.image_shape[1]], bkg_weight,dtype=numpy.float32)
                dense_weights[batch_index,plane_index,y_index,x_index] = weights
            else:
                dense_weights = numpy.full([labels.shape[0], self.image_shape[0], self.image_shape[1], 3], bkg_weight,dtype=numpy.float32)
                dense_weights[batch_index,y_index,x_index,plane_index] = weights

        if FLAGS.LOSS_BALANCE_SCHEME == "light":

            # This mode maintains the weights for everything as if they are unbalanced,
            # however it gives a mild boost to cosmic pixels, and a medium
            # boost to neutrino pixels.

            per_pixel_weight = 1./(numpy.prod(self.image_shape))

            bkg_weight = per_pixel_weight
            # Now we have the weight values, return it in the proper shape:
            # Prepare output weights:
            weights = numpy.full(values.shape, bkg_weight,dtype=numpy.float32)
            weights[values==self.COSMIC_INDEX]   = 1.5 * per_pixel_weight
            weights[values==self.NEUTRINO_INDEX] = 10  * per_pixel_weight

            if FLAGS.DATA_FORMAT == "channels_first":
                dense_weights = numpy.full([labels.shape[0], 3, self.image_shape[0], self.image_shape[1]], bkg_weight,dtype=numpy.float32)
                dense_weights[batch_index,plane_index,y_index,x_index] = weights
            else:
                dense_weights = numpy.full([labels.shape[0], self.image_shape[0], self.image_shape[1], 3], bkg_weight,dtype=numpy.float32)
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
