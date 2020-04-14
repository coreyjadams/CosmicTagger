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
        self._global_step  = 0
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


    def print(self, *argv):
        ''' Function for logging as needed.  Works correctly in distributed mode'''

        message = " ".join([ str(s) for s in argv] )

        sys.stdout.write(message + "\n")

    def initialize(self, io_only=True):
        self._initialize_io(color=0)

    def _initialize_io(self, color=None):

        self._train_data_size = self.larcv_fetcher.prepare_cosmic_sample(
            "train", self.args.file, self.args.minibatch_size, color)

        if self.args.aux_file is not None:
            self._aux_data_size = self.larcv_fetcher.prepare_cosmic_sample(
                "aux", self.args.aux_file, self.args.minibatch_size, color)


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

        self.log(log_string)

        return

    def on_step_end(self):
        pass

    def on_epoch_end(self):
        pass

    def metrics(self, metrics):
        # This function looks useless, but it is not.
        # It allows a handle to the distributed network to allreduce metrics.
        return metrics

    def stop(self):
        # Mostly, this is just turning off the io:
        # self._larcv_interface.stop()
        pass


    def batch_process(self, verbose=True):

        # Run iterations
        for self._iteration in range(FLAGS.ITERATIONS):
            if FLAGS.TRAINING and self._iteration >= FLAGS.ITERATIONS:
                self.log('Finished training (iteration %d)' % self._iteration)
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
