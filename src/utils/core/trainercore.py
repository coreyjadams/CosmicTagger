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
# from . larcvio   import larcv_fetcher

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
        else:
            sparse = False

        self.larcv_fetcher = larcv_fetcher.larcv_fetcher(
            mode        = args.mode.name.name,
            distributed = args.run.distributed,
            data_args   = args.data,
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
