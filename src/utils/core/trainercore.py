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


from src.utils import logging
# logger = logging.getLogger()
# logger.critical("TEST 1")
# logger.propogate = True
# print(logger)
# print(logger.handlers)
# logger.critical("TEST 2")
# logger.handlers[0].flush()
# exit()

import contextlib

import tensorboardX


from src.config import DataFormatKind, ModeKind, RunUnit

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
        self.rank           = None
        self._exit          = False
        self.latest_metrics = {}


        # Just for benchmarking measurements:
        self.post_one_time   = None
        self.post_two_time   = None
        self.iteration_start = None
        self.times           = []
        # if args.framework.name == "torch":
        #     sparse = args.framework.sparse
        # else:
        #     sparse = False
        #
        # if args.data.data_format == DataFormatKind.channels_first: self._channels_dim = 1
        # if args.data.data_format == DataFormatKind.channels_last : self._channels_dim = -1

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
        self.profiling_array = numpy.zeros((500,), dtype=self.profiling_dtype)
        #
        # self._log_keys = ['Average/Non_Bkg_Accuracy', 'Average/mIoU']
        # if self.args.network.classification.active:
        #     self._log_keys += ['Average/EventLabel',]
        # if self.args.network.vertex.active:
        #     self._log_keys += ['Average/VertexDetection',]
        # if self.is_training():
        #     self._log_keys.append("loss/total")
        #
        # # Copy these:
        # self._hparams_keys = [ lk for lk in  self._log_keys]
        # # Add to it
        # self._hparams_keys += ["Average/Neutrino_IoU"]
        # self._hparams_keys += ["Average/Cosmic_IoU"]
        # self._hparams_keys += ["Average/Total_Accuracy"]
        # self._hparams_keys += ["loss/segmentation"]
        # if self.args.network.classification.active:
        #     self._hparams_keys += ['loss/event_label',]
        # if self.args.network.vertex.active:
        #     self._hparams_keys += ['Average/VertexResolution',]
        #     self._hparams_keys += ['loss/vertex/detection',]
        #     self._hparams_keys += ['loss/vertex/localization',]

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

    def log(self, metrics, log_keys=[], saver=''):

        if self._global_step % self.args.mode.logging_iteration == 0:

            self._current_log_time = datetime.datetime.now()

            # Build up a string for logging:
            if log_keys != []:
                s = ", ".join(["{0}: {1:.3}".format(key, metrics[key]) for key in log_keys])
            else:
                s = ", ".join(["{0}: {1:.3}".format(key, metrics[key]) for key in metrics])

            time_string = []

            if hasattr(self, "_previous_log_time"):
            # try:
                total_images = self.args.run.minibatch_size
                images_per_second = total_images / (self._current_log_time - self._previous_log_time).total_seconds()
                time_string.append("{:.2} Img/s".format(images_per_second))

            if 'io_fetch_time' in metrics.keys():
                time_string.append("{:.2} IOs".format(metrics['io_fetch_time']))

            if 'step_time' in metrics.keys():
                time_string.append("{:.2} (Step)(s)".format(metrics['step_time']))

            if len(time_string) > 0:
                s += " (" + " / ".join(time_string) + ")"

            self._previous_log_time = self._current_log_time
            logging.getLogger("CosmicTagger""CosmicTagger").info("{} Step {} metrics: {}".format(saver, self._global_step, s))

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

        if self.rank is None or self.rank == 0:
            os.makedirs(top_dir, exist_ok=True)
            name = top_dir + "profiling_inforank_0"
        else:
            name = top_dir + f"profiling_inforank_{self.rank}"

        # This barrier enforces the root rank has made the folder before
        # anyone tries to write.
        logging.getLogger("CosmicTagger").info("Saving run profile information.")

        self.barrier()

        # If the file already exists, remove it:

        # Save the arguments too:
        if self.rank is None or self.rank == 0:
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

    def extend_profiling_array(self):
        # Resize the profiling array if needed:
        if self.profiling_index >= len(self.profiling_array) - 1:
            # Add 500 more rows:
            self.profiling_array.resize((self.profiling_index + 500))

        return

    def on_step_start(self):
        self.extend_profiling_array()
        self.profiling_array[self.profiling_index]["i"] = self._iteration
        self.profiling_array[self.profiling_index]["start"] = self.now()
        self.iteration_start = time.time()

    def on_step_end(self):
        if self.post_one_time is None:
            self.post_one_time = time.time()
        elif self.post_two_time is None:
            self.post_two_time = time.time()
        self.times.append(time.time() - self.iteration_start)
        self.profiling_index += 1

        # Checkpoint after every step
        # (Not really, this function has logic to decide if it checkpoints)
        with self.timing_context("checkpoint"):
            self.checkpoint()

    def on_epoch_start(self):
        pass

    def on_epoch_end(self):

        self._epoch += 1
        self._epoch_end = True
        with self.timing_context("checkpoint"):
            self.checkpoint()
        self._epoch_end = False


    def train_one_epoch(self, train_loader, val_loader=None, max_steps=None):
        self.on_epoch_start()
        for i, batch in enumerate(train_loader):

            # Check step end condition:
            if max_steps is not None:
                 if self._iteration > max_steps: self._exit = True

            if self._exit: break

            self.on_step_start()
            with self.timing_context("train"):
                self.train_step(batch)
            # Validate one batch if using iterations as a measure:
            if self.args.run.run_units == RunUnit.iteration and  self._iteration % self.args.run.val_iteration == 0:
                if val_loader is not None: self.validate(val_loader, max_steps=1)

            self.on_step_end()
            self._iteration += 1

        if self.args.run.run_units == RunUnit.epoch:
            # Validate an entire epoch if we're using epochs:
            if val_loader is not None: self.validate(val_loader, max_steps = None)
        self.on_epoch_end()

    def validate(self, val_loader, max_steps):

        metrics_list = []
        # Print out on every iteration or none?
        store = False
        if max_steps is not None: store=True

        for i, batch in enumerate(val_loader):
            if max_steps is not None and i >= max_steps:
                break
            with self.timing_context("val"):
                metrics_list.append(self.val_step(batch, store))


        self.finalize_metrics(metrics_list)

    def finalize_metrics(self, metrics_list):
        pass

    def batch_process(self, data_loaders, max_epochs=None, max_steps=None):

        logger = logging.getLogger("CosmicTagger")


        start = time.time()

        # This is the 'master' function, so it controls a lot

        self.profiling_index = 0

        # Start looping:

        self._iteration = 0
        self._epoch     = 0

        val_loader = data_loaders['val'] if 'val' in data_loaders else None

        if self.is_training():
            while not self._exit:
                if max_epochs is not None:
                    if self._epoch > max_epochs: self._exit = True
                self.train_one_epoch(data_loaders["train"], val_loader, max_steps)


            self.checkpoint()

        else:
            self.analyze(data_loaders["test"])

        # Check step end condition:
        if max_steps is not None:
            if self._iteration > max_steps: self._exit = True


        # # Run iterations
        # for self._iteration in range(self.args.run.iterations):
        #
        #
        #
        #     with self.timing_context("iteration"):
        #         if self.is_training() and self._iteration >= self.args.run.iterations:
        #
        #             logger.info('Finished training (iteration %d)' % self._iteration)
        #             break
        #
        #
        #



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
        if self.post_one_time is not None:
            throughput = (self._iteration - 1) * total_images_per_batch
            throughput /= (end - self.post_one_time)
            logger.info("Total time to batch process except first iteration: "
                        f"{end - self.post_one_time:.4f}"
                        f", throughput: {throughput:.4f}")
        if self.post_two_time is not None:
            throughput = (self._iteration - 2) * total_images_per_batch
            throughput /= (end - self.post_two_time)
            logger.info("Total time to batch process except first two iterations: "
                        f"{end - self.post_two_time:.4f}"
                        f", throughput: {throughput:.4f}")
        if len(self.times) > 40:
            throughput = (40) * total_images_per_batch
            throughput /= (numpy.sum(self.times[-40:]))
            logger.info("Total time to batch process last 40 iterations: "
                        f"{numpy.sum(self.times[-40:]):.4f}"
                        f", throughput: {throughput:.4f}" )
