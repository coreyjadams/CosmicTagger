import os
import sys
import time
import tempfile
import copy
import contextlib
import pickle

from collections import OrderedDict

import numpy

# larcv_fetcher can also do synthetic IO without any larcv installation
from . larcvio   import larcv_fetcher

import datetime
import pathlib


import logging
logger = logging.getLogger()

import contextlib

import tensorboardX


from src.config import DataFormatKind, ModeKind

class trainercore(object):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''



    def __init__(self, args):

        self._iteration     = 0
        self._global_step   = 0
        self.args           = args
        self._rank          = None
        self._exit          = False
        self.latest_metrics = {}

        if args.framework.name == "torch":
            sparse = args.framework.sparse
            io_dataformat = DataFormatKind.channels_first
        else:
            sparse = False
            io_dataformat = args.data.data_format
        # Intervene here to make sure we use the right data format:
        data_args = copy.copy(args.data)
        data_args.data_format = io_dataformat

        self.larcv_fetcher = larcv_fetcher.larcv_fetcher(
            mode        = args.mode.name.name,
            distributed = args.run.distributed,
            data_args   = data_args,
            sparse      = sparse,
            vtx_depth   = args.network.depth - args.network.vertex.depth )

        if args.data.data_format == DataFormatKind.channels_first: self._channels_dim = 1
        if args.data.data_format == DataFormatKind.channels_last : self._channels_dim = -1

        # Define a datatype for a profiling array.
        # It is going to be mostly 64bit timestamps for a number of points
        self.profiling_dtype = numpy.dtype([
            ("i",           numpy.int32),       # Filled in batch_process
            ("start",       "datetime64[us]"),  # Filled in batch_process
            ("iteration",   "timedelta64[us]"), # Filled in batch_process
            ("train",       "timedelta64[us]"), # Filled in batch_process
            ("val",         "timedelta64[us]"), # Filled in batch_process
            ("io",          "timedelta64[us]"), # Filled in train_step + val_step in both TF/Torch
            ("forward",     "timedelta64[us]"),
            ("backward",    "timedelta64[us]"),
            ("checkpoint",  "timedelta64[us]"), # Filled in batch_process
            ("loss",        "timedelta64[us]"),
            ("summary",     "timedelta64[us]"),
            ("log",         "timedelta64[us]"),
            ("optimizer",   "timedelta64[us]"),
            ("metrics",     "timedelta64[us]"),
        ])

        # Create the baseline array:
        self.profiling_array = numpy.zeros((args.run.iterations,), dtype=self.profiling_dtype)

        self._log_keys = ['Average/Non_Bkg_Accuracy', 'Average/mIoU']
        if self.args.network.classification.active:
            self._log_keys += ['Average/EventLabel',]
        if self.args.network.vertex.active:
            self._log_keys += ['Average/VertexDetection',]
        if self.is_training():
            self._log_keys.append("loss/total")

        # Copy these:
        self._hparams_keys = [ lk for lk in  self._log_keys]
        # Add to it
        self._hparams_keys += ["Average/Neutrino_IoU"]
        self._hparams_keys += ["Average/Cosmic_IoU"]
        self._hparams_keys += ["Average/Total_Accuracy"]
        self._hparams_keys += ["loss/segmentation"]
        if self.args.network.classification.active:
            self._hparams_keys += ['loss/event_label',]
        if self.args.network.vertex.active:
            self._hparams_keys += ['Average/VertexResolution',]
            self._hparams_keys += ['loss/vertex/detection',]
            self._hparams_keys += ['loss/vertex/localization',]

    def now(self):
        return numpy.datetime64(datetime.datetime.now())

    def is_training(self):
        return self.args.mode.name == ModeKind.train

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
            if self.is_training():
                logger.warning("WARNING: Aux file does not exist.  Setting to None for training")
                self.args.data.aux_file = ""
            else:
                # In inference mode, we are creating the aux file.  So we need to check
                # that the directory exists.  Otherwise, no writing.
                if not aux_f.parent.exists():
                    logger.warning("WARNING: Aux file's directory does not exist.")
                    self.args.data.aux_file = ""
                elif self.args.data.aux_file is None or str(self.args.data.aux_file).lower() == "none":
                    logger.warning("WARNING: no aux file set, so not writing inference results.")
                    self.args.data.aux_file = ""


        self._train_data_size = self.larcv_fetcher.prepare_cosmic_sample(
            "train", f, self.args.run.minibatch_size, color)

        if not self.args.data.synthetic and self.args.data.aux_file != "":
            if self.is_training():
                # Fetching data for on the fly testing:
                self._aux_data_size = self.larcv_fetcher.prepare_cosmic_sample(
                    "aux", aux_f, self.args.run.minibatch_size, color)
            elif self.args.mode == ModeKind.inference:
                self._aux_data_size = self.larcv_fetcher.prepare_writer(
                    input_file = f, output_file = str(aux_f))


    def build_lr_schedule(self, learning_rate_schedule = None):
        # Define the learning rate sequence:



        if learning_rate_schedule is None:
            learning_rate_schedule = {
                'warm_up' : {
                    'function'      : 'linear',
                    'start'         : 1e-4,
                    'n_epochs'      : 1,
                    'initial_rate'  : 0.00001,
                },
                'flat' : {
                    'function'      : 'flat',
                    'start'         : 1,
                    'n_epochs'      : 6,
                },
                'decay' : {
                    'function'      : 'decay',
                    'start'         : 7,
                    'n_epochs'      : 5,
                        'floor'         : 0.1*self.args.mode.optimizer.learning_rate,
                    'decay_rate'    : 0.99999
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
            x  / self.args.run.epoch,
            [c(x  / self.args.run.epoch) for c in cond_list], func_list)


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

    def inference_report(self):
        pass

    def write_profiling_info(self):

        top_dir = self.args.output_dir + "/profiles/"

        if self._rank is None or self._rank == 0:
            os.makedirs(top_dir, exist_ok=True)
            name = top_dir + "profiling_info_rank_0"
        else:
            name = top_dir + f"profiling_info_rank_{self._rank}"

        # This barrier enforces the root rank has made the folder before
        # anyone tries to write.
        logger.info("Saving run profile information.")

        self.barrier()

        # If the file already exists, remove it:

        # Save the arguments too:
        if self._rank is None or self._rank == 0:
            with open(top_dir + "args.pkl", "wb") as f:
                pickle.dump(self.args, f, protocol=pickle.HIGHEST_PROTOCOL)

        numpy.save(name, self.profiling_array)

    def flatten(self, dictionary_object, prefix=""):

        new_dict = {}
        for key in dictionary_object:
            new_key = prefix + key
            if type(dictionary_object[key]) == type(dictionary_object):
                new_dict.update(self.flatten(dictionary_object[key], prefix=new_key + "/"))
            else:
                if hasattr(dictionary_object[key], "value"):
                    new_dict[new_key] = str(dictionary_object[key].value)
                else:
                    new_dict[new_key] = dictionary_object[key]

        return new_dict



    def store_parameters(self, metrics):

        return

    def exit(self):
        self._exit = True
        return

    def barrier(self): return

    @contextlib.contextmanager
    def timing_context(self, key):
        start = self.now()
        yield
        self.profiling_array[self.profiling_index][key] = self.now() - start


    def close_savers(self):
        pass

    def batch_process(self):

        start = time.time()
        post_one_time = None
        post_two_time = None

        times = []

        # This is the 'master' function, so it controls a lot


        self.profiling_index = 0

        # Run iterations
        for self._iteration in range(self.args.run.iterations):

            if self._exit: break

            # Resize the profiling array if needed:
            if self.profiling_index > len(self.profiling_array) - 1:
                # Add 500 more rows:
                self.profiling_array.resize((self.profiling_index + 500))

            self.profiling_array[self.profiling_index]["i"] = self._iteration
            self.profiling_array[self.profiling_index]["start"] = self.now()
            with self.timing_context("iteration"):
                iteration_start = time.time()
                if self.is_training() and self._iteration >= self.args.run.iterations:

                    logger.info('Finished training (iteration %d)' % self._iteration)
                    self.checkpoint()
                    break


                if self.is_training():
                    with self.timing_context("val"):
                        self.val_step()
                    with self.timing_context("train"):
                        self.train_step()
                    with self.timing_context("checkpoint"):
                        self.checkpoint()
                else:
                    self.ana_step()

                if post_one_time is None:
                    post_one_time = time.time()
                elif post_two_time is None:
                    post_two_time = time.time()
                times.append(time.time() - iteration_start)
            self.profiling_index += 1


        self.write_profiling_info()

        end = time.time()

        if self.args.data.synthetic and self.args.run.distributed:
            try:
                total_images_per_batch = self.args.run.minibatch_size * self._size
            except:
                total_images_per_batch = self.args.run.minibatch_size
        else:
            total_images_per_batch = self.args.run.minibatch_size


        if self.args.mode.name == ModeKind.inference:
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
