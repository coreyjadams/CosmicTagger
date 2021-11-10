import os
import sys
import time
import tempfile
import copy

from collections import OrderedDict

import numpy

# larcv_fetcher can also do synthetic IO without any larcv installation
from . larcvio   import larcv_fetcher

import datetime
import pathlib


import logging
logger = logging.getLogger("cosmictagger")

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
        if args.framework.name == "torch":
            sparse = args.framework.sparse
        else:
            sparse = False

        self.larcv_fetcher = larcv_fetcher.larcv_fetcher(
            mode        = args.mode.name,
            distributed = args.run.distributed,
            downsample  = args.data.downsample,
            dataformat  = args.data.data_format,
            synthetic   = args.data.synthetic,
            sparse      = sparse )

        if args.data.data_format == "channels_first": self._channels_dim = 1
        if args.data.data_format == "channels_last" : self._channels_dim = -1



    def initialize(self, io_only=True):
        self._initialize_io(color=0)

    def _initialize_io(self, color=None):

        if self.args.mode == "build_net": return


        if not self.args.data.synthetic:
            f = pathlib.Path(self.args.data.data_directory + self.args.data.file)
            aux_f = pathlib.Path(self.args.data.data_directory + self.args.data.aux_file)
        else:
            f = None; aux_f = None

        # Check that the training file exists:
        if not self.args.data.synthetic and not f.exists():
            raise Exception(f"Can not continue with file {f} - does not exist.")
        if not self.args.data.synthetic and not aux_f.exists():
            if self.args.mode.name == "train":
                logger.warning("WARNING: Aux file does not exist.  Setting to None for training")
                self.args.data.aux_file = None
            else:
                # In inference mode, we are creating the aux file.  So we need to check
                # that the directory exists.  Otherwise, no writing.
                if not aux_f.parent.exists():
                    logger.warning("WARNING: Aux file's directory does not exist.")
                    self.args.data.aux_file = None
                elif self.args.data.aux_file is None or str(self.args.data.aux_file).lower() == "none":
                    logger.warning("WARNING: no aux file set, so not writing inference results.")
                    self.args.data.aux_file = None


        self._train_data_size = self.larcv_fetcher.prepare_cosmic_sample(
            "train", f, self.args.run.minibatch_size, color)

        if aux_f is not None:
            if self.args.mode.name == "train":
                # Fetching data for on the fly testing:
                self._aux_data_size = self.larcv_fetcher.prepare_cosmic_sample(
                    "aux", aux_f, self.args.run.minibatch_size, color)
            elif self.args.mode == "inference":
                self._aux_data_size = self.larcv_fetcher.prepare_writer(
                    input_file = f, output_file = str(aux_f))


    def build_lr_schedule(self, learning_rate_schedule = None):
        # Define the learning rate sequence:

        if learning_rate_schedule is None:
            learning_rate_schedule = {
                'warm_up' : {
                    'function'      : 'linear',
                    'start'         : 0,
                    'n_epochs'      : 1,
                    'initial_rate'  : 0.00001,
                },
                'flat' : {
                    'function'      : 'flat',
                    'start'         : 1,
                    'n_epochs'      : 20,
                },
                'decay' : {
                    'function'      : 'decay',
                    'start'         : 21,
                    'n_epochs'      : 4,
                        'floor'         : 0.0001,
                    'decay_rate'    : 0.999
                },
            }

        # one_cycle_schedule = {
        #     'ramp_up' : {
        #         'function'      : 'linear',
        #         'start'         : 0,
        #         'n_epochs'      : 10,
        #         'initial_rate'  : 0.00001,
        #         'final_rate'    : 0.001,
        #     },
        #     'ramp_down' : {
        #         'function'      : 'linear',
        #         'start'         : 10,
        #         'n_epochs'      : 10,
        #         'initial_rate'  : 0.001,
        #         'final_rate'    : 0.00001,
        #     },
        #     'decay' : {
        #         'function'      : 'decay',
        #         'start'         : 20,
        #         'n_epochs'      : 5,
        #         'rate'          : 0.00001
        #         'floor'         : 0.00001,
        #         'decay_rate'    : 0.99
        #     },
        # }
        # learning_rate_schedule = one_cycle_schedule

        # We build up the functions we need piecewise:
        func_list = []
        cond_list = []

        for i, key in enumerate(learning_rate_schedule):

            # First, create the condition for this stage
            start    = learning_rate_schedule[key]['start']
            length   = learning_rate_schedule[key]['n_epochs']

            if i +1 == len(learning_rate_schedule):
                # Make sure the condition is open ended if this is the last stage
                condition = lambda x, s=start, l=length: x >= s
            else:
                # otherwise bounded
                condition = lambda x, s=start, l=length: x >= s and x < s + l


            if learning_rate_schedule[key]['function'] == 'linear':

                initial_rate = learning_rate_schedule[key]['initial_rate']
                if 'final_rate' in learning_rate_schedule[key]: final_rate = learning_rate_schedule[key]['final_rate']
                else: final_rate = self.args.mode.optimizer.learning_rate

                function = lambda x, s=start, l=length, i=initial_rate, f=final_rate : numpy.interp(x, [s, s + l] ,[i, f] )

            elif learning_rate_schedule[key]['function'] == 'flat':
                if 'rate' in learning_rate_schedule[key]: rate = learning_rate_schedule[key]['rate']
                else: rate = self.args.mode.optimizer.learning_rate

                function = lambda x : rate

            elif learning_rate_schedule[key]['function'] == 'decay':
                decay    = learning_rate_schedule[key]['decay_rate']
                floor    = learning_rate_schedule[key]['floor']
                if 'rate' in learning_rate_schedule[key]: rate = learning_rate_schedule[key]['rate']
                else: rate = self.args.mode.optimizer.learning_rate

                function = lambda x, s=start, d=decay, f=floor: (rate-f) * numpy.exp( -(d * (x - s))) + f

            cond_list.append(condition)
            func_list.append(function)

        self.lr_calculator = lambda x: numpy.piecewise(
            x * (self.args.run.minibatch_size / self._train_data_size),
            [c(x * (self.args.run.minibatch_size / self._train_data_size)) for c in cond_list], func_list)


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

        logger.info(log_string)

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

    def close_savers(self):
        pass

    def inference_report(self):
        pass

    def batch_process(self):


        start = time.time()
        post_one_time = None
        post_two_time = None

        times = []

        # This is the 'master' function, so it controls a lot

        # Run iterations
        for self._iteration in range(self.args.run.iterations):
            iteration_start = time.time()
            if self.args.mode.name == "train" and self._iteration >= self.args.run.iterations:

                logger.info('Finished training (iteration %d)' % self._iteration)
                self.checkpoint()
                break


            if self.args.mode.name == "train":
                self.val_step()
                self.train_step()
                self.checkpoint()
            else:
                self.ana_step()

            if post_one_time is None:
                post_one_time = time.time()
            elif post_two_time is None:
                post_two_time = time.time()
            times.append(time.time() - iteration_start)
        self.close_savers()

        end = time.time()

        if self.args.data.synthetic and self.args.run.distributed:
            try:
                total_images_per_batch = self.args.run.minibatch_size * self._size
            except:
                total_images_per_batch = self.args.run.minibatch_size
        else:
            total_images_per_batch = self.args.run.minibatch_size


        if self.args.mode.name == "inference":
            self.inference_report()

        logger.info(f"Total time to batch_process: {end - start:.4f}")
        if post_one_time is not None:
            throughput = (self.args.run.iterations - 1) * total_images_per_batch
            throughput /= (end - post_one_time)
            logger.info("Total time to batch process except first iteration: "
                        f"{end - post_one_time:.4f}"
                        f", throughput: {throughput:.4f}")
        if post_two_time is not None:
            throughput = (self.args.run.iterations - 2) * total_images_per_batch
            throughput /= (end - post_two_time)
            logger.info("Total time to batch process except first two iterations: "
                        f"{end - post_two_time:.4f}"
                        f", throughput: {throughput:.4f}")
        if len(times) > 40:
            throughput = (40) * total_images_per_batch
            throughput /= (numpy.sum(times[-40:]))
            logger.info("Total time to batch process last 40 iterations: "
                        f"{numpy.sum(times[-40:]):.4f}"
                        f", throughput: {throughput:.4f}" )
