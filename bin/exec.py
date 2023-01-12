#!/usr/bin/env python
import os,sys,signal
import time
import pathlib
import logging
from logging import handlers

import numpy

# For configuration:
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.experimental import compose, initialize
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import configure_log

hydra.output_subdir = None

#############################

# Add the local folder to the import path:
network_dir = os.path.dirname(os.path.abspath(__file__))
network_dir = os.path.dirname(network_dir)
sys.path.insert(0,network_dir)

from src.config import Config, RunUnit
from src.config.mode import ModeKind

from src.utils.io import create_larcv_dataset

import atexit

class exec(object):

    def __init__(self, config):

        self.args = config

        rank = self.init_mpi()


        # Add to the output dir:
        self.args.output_dir += f"/{self.args.framework.name}/"
        self.args.output_dir += f"/{self.args.network.name}/"
        self.args.output_dir += f"/{self.args.run.id}/"

        # Create the output directory if needed:
        if rank == 0:
            outpath = pathlib.Path(self.args.output_dir)
            outpath.mkdir(exist_ok=True, parents=True)

        self.configure_logger(rank)

        self.validate_arguments()

        # Print the command line args to the log file:
        logger = logging.getLogger()
        logger.info("Dumping launch arguments.")
        logger.info(sys.argv)
        logger.info(self.__str__())

        logger.info("Configuring Datasets.")
        self.datasets = self.configure_datasets()
        logger.info("Data pipeline ready.")


    def run(self):
        if self.args.mode.name == ModeKind.train:
            self.train()
        if self.args.mode.name == ModeKind.iotest:
            self.iotest()
        if self.args.mode.name == ModeKind.inference:
            self.inference()

    def exit(self):
        if hasattr(self, "trainer"):
            self.trainer.exit()

    def init_mpi(self):
        if not self.args.run.distributed:
            return 0
        else:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            return comm.Get_rank()


    def configure_datasets(self):
        """
        This function creates the non-framework iterable datasets used in this app.

        They get converted to framework specific tools, if needed, in the 
        framework specific code.
        """

        # Check if we need vertex or eventID info:

        if self.args.network.vertex.active:
            vertex_depth = self.args.network.vertex.depth
            event_id = True

        if self.args.network.classification.active:
            event_id = True

        if self.args.data.synthetic:
            datasets = { 
                "synthetic" : create_larcv_dataset(
                    data_args    = self.args.data, 
                    batch_size   = self.args.run.minibatch_size, 
                    input_file   = None,
                    distributed  = False, 
                    event_id     = event_id, 
                    vertex_depth = vertex_depth, 
                    sparse       = False) 
            }
        else:
            datasets = {
                name : create_larcv_dataset(
                    data_args    = self.args.data, 
                    batch_size   = self.args.run.minibatch_size, 
                    input_file   = getattr(self.args.data.paths, name),
                    distributed  = self.args.run.distributed, 
                    event_id     = event_id, 
                    vertex_depth = vertex_depth, 
                    sparse       = self.args.framework.sparse) 
                for name in self.args.data.paths.active
            }

        return datasets


    def configure_logger(self, rank):

        logger = logging.getLogger()
        # Create a handler for STDOUT, but only on the root rank.
        # If not distributed, we still get 0 passed in here.
        if rank == 0:
            stream_handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            stream_handler.setFormatter(formatter)
            handler = handlers.MemoryHandler(capacity = 0, target=stream_handler)
            logger.addHandler(handler)

            # Add a file handler too:
            log_file = self.args.output_dir + "/process.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            file_handler = handlers.MemoryHandler(capacity=10, target=file_handler)
            logger.addHandler(file_handler)

            logger.setLevel(logging.INFO)
        else:
            # in this case, MPI is available but it's not rank 0
            # create a null handler
            handler = logging.NullHandler()
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)



    def train(self):

        logger = logging.getLogger()

        logger.info("Running Training")

        self.make_trainer()

        self.trainer.initialize()
        self.trainer.batch_process()


    def iotest(self):

        # self.make_trainer()
        logger = logging.getLogger()

        logger.info("Running IO Test")


        # self.trainer.initialize(io_only=True)

        if self.args.run.distributed:
            from mpi4py import MPI
            rank = MPI.COMM_WORLD.Get_rank()
        else:
            rank = 0




        for key, dataset in self.datasets.items():

            global_start = time.time()
            total_reads = 0 


            # Determine the stopping point:
            if self.args.run.run_units == RunUnit.epoch:
                break_i = self.args.run.run_length * len(dataset)
            else:
                break_i = self.args.run.run_length

            start = time.time()
            for i, minibatch in enumerate(dataset):
                # mb = self.trainer.larcv_fetcher.fetch_next_batch("train", force_pop=True)

                end = time.time()
                if i >= break_i: break

                logger.info(f"{i}: Time to fetch a minibatch of data: {end - start:.2f}s")
                start = time.time()
                total_reads += 1

            total_time = time.time() - global_start
            images_read = total_reads * self.args.run.minibatch_size
            logger.info(f"{key} - Total IO Time: {total_time:.2f}s")
            logger.info(f"{key} - Total images read per batch: {self.args.run.minibatch_size}")
            logger.info(f"{key} - Average Image IO Throughput: { images_read / total_time:.3f}")

    def make_trainer(self):


        if 'environment_variables' in self.args.framework:
            for env in self.args.framework.environment_variables.keys():
                os.environ[env] = self.args.framework.environment_variables[env]

        if self.args.mode.name == ModeKind.iotest:
            from src.utils.core import trainercore
            self.trainer = trainercore.trainercore(self.args)
            return

        if self.args.framework.name == "tensorflow":

            import logging
            logging.getLogger('tensorflow').setLevel(logging.FATAL)


            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

            # Import tensorflow and see what the version is.
            import tensorflow as tf

            if tf.__version__.startswith("2"):
                if self.args.run.distributed:
                    from src.utils.tensorflow2 import distributed_trainer
                    self.trainer = distributed_trainer.distributed_trainer(self.args)
                else:
                    from src.utils.tensorflow2 import trainer
                    self.trainer = trainer.tf_trainer(self.args)
            else:
                if self.args.run.distributed:
                    from src.utils.tensorflow1 import distributed_trainer
                    self.trainer = distributed_trainer.distributed_trainer(self.args)
                else:
                    from src.utils.tensorflow1 import trainer
                    self.trainer = trainer.tf_trainer(self.args)

        elif self.args.framework.name == "torch":
            if self.args.run.distributed:
                from src.utils.torch import distributed_trainer
                self.trainer = distributed_trainer.distributed_trainer(self.args)
            else:
                from src.utils.torch import trainer
                self.trainer = trainer.torch_trainer(self.args)


    def inference(self):


        logger = logging.getLogger()

        logger.info("Running Inference")
        logger.info(self.__str__())

        self.make_trainer()

        self.trainer.initialize()
        self.trainer.batch_process()

    def dictionary_to_str(self, in_dict, indentation = 0):
        substr = ""
        for key in sorted(in_dict.keys()):
            if type(in_dict[key]) == DictConfig or type(in_dict[key]) == dict:
                s = "{none:{fill1}{align1}{width1}}{key}: \n".format(
                        none="", fill1=" ", align1="<", width1=indentation, key=key
                    )
                substr += s + self.dictionary_to_str(in_dict[key], indentation=indentation+2)
            else:
                if hasattr(in_dict[key], "name"): attr = in_dict[key].name
                else: attr = in_dict[key]
                s = '{none:{fill1}{align1}{width1}}{message:{fill2}{align2}{width2}}: {attr}\n'.format(
                   none= "",
                   fill1=" ",
                   align1="<",
                   width1=indentation,
                   message=key,
                   fill2='.',
                   align2='<',
                   width2=30-indentation,
                   attr = attr,
                )
                substr += s
        return substr

    def __str__(self):

        s = "\n\n-- CONFIG --\n"
        substring = s +  self.dictionary_to_str(self.args)

        return substring




    def validate_arguments(self):

        from src.config.data import DataFormatKind

        logger = logging.getLogger()

        if self.args.framework.name == "torch":
            # In torch, only option is channels first:
            if self.args.data.data_format == DataFormatKind.channels_last:
                logger.warning("Torch requires channels_first, switching automatically")
                self.args.data.data_format = DataFormatKind.channels_first

        elif self.args.framework.name == "tensorflow":
            if self.args.mode.name == ModeKind.train:
                if self.args.mode.quantization_aware:
                    logger.error("Quantization aware training not implemented in tensorflow.")

        self.args.network.data_format = self.args.data.data_format.name




@hydra.main(version_base=None, config_path="../src/config/recipes/", config_name="config")
def main(cfg : OmegaConf) -> None:

    s = exec(cfg)
    atexit.register(s.exit)

    s.run()

if __name__ == '__main__':
    #  Is this good practice?  No.  But hydra doesn't give a great alternative
    import sys
    if "--help" not in sys.argv and "--hydra-help" not in sys.argv:
        sys.argv += [
            'hydra/job_logging=disabled',
            'hydra.output_subdir=null',
            'hydra.job.chdir=False',
            'hydra.run.dir=.',
            'hydra/hydra_logging=disabled',
        ]
    main()
