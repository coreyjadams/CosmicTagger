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


class exec(object):

    def __init__(self, config):

        self.args = config
        
        rank = self.init_mpi()

        self.configure_logger(rank)

        self.validate_arguments()

        for k,v in  logging.Logger.manager.loggerDict.items()  :
            print('+ [%s] {%s} ' % (str.ljust( k, 20)  , str(v.__class__)[8:-2]) ) 
            if not isinstance(v, logging.PlaceHolder):
                for h in v.handlers:
                    print('     +++',str(h.__class__)[8:-2] )


        if config.mode.name == "train":
            self.train()
        if config.mode.name == "iotest":
            self.iotest()

    def init_mpi(self):
        if not self.args.run.distributed:
            return 0
        else:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            return comm.Get_rank()


    def configure_logger(self, rank):

        logger = logging.getLogger()

        # Create a handler for STDOUT, but only on the root rank.
        # If not distributed, we still get 0 passed in here.
        if rank == 0:
            stream_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            stream_handler.setFormatter(formatter)
            handler = handlers.MemoryHandler(capacity = 0, target=stream_handler)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
        else:
            # in this case, MPI is available but it's not rank 0
            # create a null handler
            handler = logging.NullHandler()
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)


    def train(self):

        logger = logging.getLogger("cosmictagger")

        logger.info("Running Training")
        logger.info(self.__str__())

        self.make_trainer()

        self.trainer.initialize()
        self.trainer.batch_process()


    def add_network_parser(self, parser):

        # Add the uresnet configuration:
        from src.networks.config import UResNetConfig
        UResNetConfig().build_parser(parser)


    def add_network_parsers(self, parser):
        # Here, we define the networks available.  In io test mode, used to determine what the IO is.
        network_parser = parser.add_subparsers(
            title          = "Networks",
            dest           = "network",
            description    = 'Which network architecture to use.')



    def iotest(self):

        self.make_trainer()
        logger = logging.getLogger("cosmictagger")

        logger.info("Running IO Test")
        logger.info(self.__str__())


        self.trainer.initialize(io_only=True)

        if self.args.distributed:
            from mpi4py import MPI
            rank = MPI.COMM_WORLD.Get_rank()
        else:
            rank = 0

        # label_stats = numpy.zeros((36,))
        global_start = time.time()
        time.sleep(0.1)
        for i in range(self.args.run.iterations):
            start = time.time()
            mb = self.trainer.larcv_fetcher.fetch_next_batch("train", force_pop=True)

            end = time.time()

            logger.info(f"{i}: Time to fetch a minibatch of data: {end - start:.2f}s")

        logger.info(f"Total IO Time: {time.time() - global_start:.2f}s")


    def make_trainer(self):


        if self.args.mode.name == "iotest":
            from src.utils.core import trainercore
            self.trainer = trainercore.trainercore(self.args)
            return

        if self.args.framework.name == "tensorflow":

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
        self.parser = argparse.ArgumentParser(
            description     = 'Run Network Inference',
            formatter_class = argparse.ArgumentDefaultsHelpFormatter)

        self.add_io_arguments(self.parser)
        self.add_core_configuration(self.parser)
        self.add_shared_training_arguments(self.parser)

        self.add_network_parser(self.parser)

        self.args = self.parser.parse_args(sys.argv[2:])
        self.args.training = False
        self.args.mode = "inference"


        self.make_trainer()
        logger = logging.getLogger("cosmictagger")

        logger.info("Running Inference")
        logger.info(self.__str__())

        self.trainer.initialize()
        self.trainer.batch_process()


    def build_net(self):
        self.parser = argparse.ArgumentParser(
            description     = 'Build network and return parameters',
            formatter_class = argparse.ArgumentDefaultsHelpFormatter)

        self.add_io_arguments(self.parser)
        self.add_core_configuration(self.parser)
        self.add_shared_training_arguments(self.parser)

        self.add_network_parser(self.parser)

        self.args = self.parser.parse_args(sys.argv[2:])
        self.args.training = False
        self.args.mode = "inference"


        self.make_trainer()

        logger = logging.getLogger("cosmictagger")
        logger.info("Running Inference")

        # self.trainer.initialize()
        # self.trainer.print()
        self.trainer.init_network()
        self.trainer.print_network_info(verbose=True)
        self.trainer.print(F"NUMBER_OF_PARAMETERS: {self.trainer.n_parameters()}")
        # self.trainer.batch_process()


    def dictionary_to_str(self, in_dict, indentation = 0):
        substr = ""
        for key in sorted(in_dict.keys()):
            if type(in_dict[key]) == DictConfig or type(in_dict[key]) == dict:
                s = "{none:{fill1}{align1}{width1}}{key}: \n".format(
                        none="", fill1=" ", align1="<", width1=indentation, key=key
                    )
                substr += s + self.dictionary_to_str(in_dict[key], indentation=indentation+2)
            else:
                s = '{none:{fill1}{align1}{width1}}{message:{fill2}{align2}{width2}}: {attr}\n'.format(
                   none= "",
                   fill1=" ",
                   align1="<",
                   width1=indentation,
                   message=key,
                   fill2='.',
                   align2='<',
                   width2=30-indentation,
                   attr = in_dict[key],
                )
                substr += s
        return substr

    def __str__(self):

        s = "\n\n-- CONFIG --\n"
        substring = s +  self.dictionary_to_str(self.args)

        return substring




    def validate_arguments(self):


        if self.args.framework.name == "torch":
            # In torch, only option is channels first:
            if self.args.data.data_format == "channels_last":
                print("Torch requires channels_first, switching automatically")
                self.args.data.data_format = "channels_first"



@hydra.main(config_path="../src/config", config_name="config")
def main(cfg : OmegaConf) -> None:



    s = exec(cfg)


if __name__ == '__main__':
    #  Is this good practice?  No.  But hydra doesn't give a great alternative
    import sys
    sys.argv += ['hydra.run.dir=.', 'hydra/job_logging=disabled']
    main()