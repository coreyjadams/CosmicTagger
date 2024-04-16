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
        self._epoch_end = False


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
        # Store the metrics per iteration into a file, (though only if rank == 0)
        self.metric_files = {}

    def __del__(self):
        for key in self.metric_files.keys():
            self.metric_files[key].close()

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

    def write_metrics(self, metrics, kind, step):
        '''
        Write the metrics into a csv file.
        '''
        if kind not in self.metric_files:
            # Initialize the file:
            fname = f"{self.args.output_dir}/{kind}_metrics.csv"
            self.metric_files[kind] = open(fname, 'w')
            # Dump the header in:
            self.metric_files[kind].write("step,"+",".join(metrics.keys())+"\n")

        # Write the metrics in:
        values = [ f"{v:.5f}" for v in metrics.values()]
        self.metric_files[kind].write(f"{step}," + ",".join(values)+"\n")
        self.metric_files[kind].flush()
        
    def log(self, metrics, log_keys=[], saver=''):


        step = int(self._global_step)
        if step % self.args.mode.logging_iteration == 0:

            self._current_log_time = datetime.datetime.now()

            # Build up a string for logging:
            if log_keys != []:
                s = ", ".join([f"{key}: {metrics[key]:.3}" for key in log_keys])
            else:
                s = ", ".join([f"{key}: {metrics[key]:.3}" for key in metrics])

            time_string = []

            if hasattr(self, "_previous_log_time"):
            # try:
                total_images = self.args.run.minibatch_size
                images_per_second = total_images / (self._current_log_time - self._previous_log_time).total_seconds()
                time_string.append(f"{images_per_second:.2} Img/s")

            if 'io_fetch_time' in metrics.keys():
                time_string.append(f"{metrics['io_fetch_time']:.2} IOs")

            if 'step_time' in metrics.keys():
                time_string.append(f"{metrics['step_time']:.2} (Step)(s)")

            if len(time_string) > 0:
                s += " (" + " / ".join(time_string) + ")"

            self._previous_log_time = self._current_log_time
            logging.getLogger("CosmicTagger").info(f"{saver} Step {step} metrics: {s}".format(s))

        self.write_metrics(metrics, saver, step)

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
        
        io_start_time = datetime.datetime.now()
        for i, batch in enumerate(train_loader):
            io_fetch_time =  (datetime.datetime.now() - io_start_time).total_seconds()

            # Check step end condition:
            if max_steps is not None:
                 if self._iteration > max_steps: self._exit = True

            if self._exit: break

            self.on_step_start()
            with self.timing_context("train"):
                metrics = self.train_step(batch)
            metrics['io_fetch_time'] =  io_fetch_time


            with self.timing_context("log"):
                self.log(metrics, self.log_keys, saver="train")

            # Validate one batch if:
            if self._iteration % self.args.run.val_iteration == 0:
                if val_loader is not None: self.validate(val_loader, max_steps=1)

            self.on_step_end()
            self._iteration += 1

            io_start_time = datetime.datetime.now()

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

    def analyze(self, data_loader, max_steps):
        
        
        for i, batch in enumerate(data_loader):
            if max_steps is not None and i >= max_steps: break

            self.on_step_start()
            self.ana_step(batch)
            self.on_step_end()


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
            # First, select the target data loader for inference, only
            # one used at a time:
            if 'test' in data_loaders:
                dl = data_loaders["test"]
            elif 'val' in data_loaders:
                logger.info("Performing inference on the validation set.")
                dl = data_loaders["val"]
            elif 'train' in data_loaders:
                logger.info("Performing inference on the training set.")
                dl = data_loaders["train"]
            
            # we use the number of iterations as max steps
            # if the run_units are iterations, otherwise
            # The length of the dataset * num_epochs
            
            if self.args.run.run_units == RunUnit.iteration:
                max_steps = self.args.run.run_length
            else:
                max_steps = self.args.run.run_length * len(dl)

            self.analyze(dl, max_steps)
            


        # Check step end condition:
        if max_steps is not None:
            if self._iteration > max_steps: self._exit = True





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
