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

# For determined:
import determined as det
import torch
import torch.distributed as dist

# Add the local folder to the import path:
network_dir = os.path.dirname(os.path.abspath(__file__))
network_dir = os.path.dirname(network_dir)
sys.path.insert(0,network_dir)

from src.config import Config, ComputeMode
from src.config.mode import ModeKind

class exec(object):

    def __init__(self, config, determined_context, determined_info):

        self.args = config
        self.determined_context = determined_context
        self.determined_info = determined_info

        rank = self.determined_context.distributed.get_rank()

        # Create the output directory if needed:
        if rank == 0:
            outpath = pathlib.Path(self.args.output_dir)
            outpath.mkdir(exist_ok=True, parents=True)

        self.configure_logger(rank)

        self.validate_arguments()

        # Print the command line args to the log file:
        logger = logging.getLogger()
        if rank == 0:
            logger.info("Dumping launch arguments.")
            logger.info(sys.argv)


        if config.mode.name == ModeKind.train:
            self.train()


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

        logger = logging.getLogger("cosmictagger")

        logger.info("Running Training")
        if self.determined_context.distributed.get_rank() == 0:
            logger.info(self.__str__())

        self.make_trainer()

        self.trainer.initialize()
        self.trainer.batch_process()



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
                from src.utils.torch import det_distributed_trainer
                self.trainer = det_distributed_trainer.det_distributed_trainer(self.determined_context, self.determined_info, self.args)
            else:
                from src.utils.torch import trainer
                self.trainer = trainer.torch_trainer(self.args)



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
                if self.args.run.compute_mode == ComputeMode.GPU:
                    logger.warning("CUDA Torch requires channels_first, switching automatically")
                    self.args.data.data_format = DataFormatKind.channels_first

        elif self.args.framework.name == "tensorflow":
            if self.args.mode.name == ModeKind.train:
                if self.args.mode.quantization_aware:
                    logger.error("Quantization aware training not implemented in tensorflow.")

        self.args.network.data_format = self.args.data.data_format.name


def merge_config_from_determined(cfg):

    import json
    hparams = json.loads(os.environ["DET_HPARAMS"])
    # iterate over overriden hyperparams and update cfg
    if "override_config" in hparams:
        updated_values = OmegaConf.create(dict(hparams["override_config"]))
        cfg = OmegaConf.merge(cfg, updated_values)
        return cfg
            

@hydra.main(config_path="../src/config", config_name="config")
#@hydra.main(version_base=None, config_path="../src/config", config_name="config")
def main(cfg : OmegaConf) -> None:

    # override config(args) from our yaml files
    cfg = merge_config_from_determined(cfg)
    
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    info = det.get_cluster_info()
    slots_per_node = len(info.slot_ids)
    num_nodes = len(info.container_addrs)
    cross_rank = info.container_rank
    cross_size = int(size / slots_per_node)
    local_rank = int(rank % slots_per_node)

    # bootstrapping for torch dist
    C10D_PORT = str(29400)
    chief_ip = info.container_addrs[0]
    os.environ['MASTER_ADDR'] = chief_ip
    os.environ['MASTER_PORT'] = C10D_PORT


    distributed = det.core.DistributedContext(
        rank=rank,
        size=size,
        local_rank=local_rank,
        local_size=slots_per_node,
        cross_rank=cross_rank,
        cross_size=num_nodes,
        chief_ip=chief_ip,
    )
    with det.core.init(distributed=distributed) as determined_context:
        world_size = determined_context.distributed.size
        num_gpus_per_machine = determined_context.distributed.local_size
        machine_rank = determined_context.distributed.cross_rank
        local_rank = determined_context.distributed.local_rank

        print (f"world_size = {world_size}, mpi_rank = {rank},  rank = {determined_context.distributed.rank} num_gpus_per_machine = {num_gpus_per_machine}, machine_rank = {machine_rank}, local_rank = {local_rank}")


        # TBR
        MASTER_PORT_NUM = str(29400)
        os.environ['MASTER_ADDR'] = determined_context.distributed._chief_ip
        os.environ['MASTER_PORT'] = MASTER_PORT_NUM
        print (f'set os env MASTER_ADDR = {determined_context.distributed._chief_ip}')
        print (f'set os env MASTER_PORT = {MASTER_PORT_NUM}')

        determined_info = det.get_cluster_info()

        s = exec(cfg, determined_context, determined_info)


if __name__ == '__main__':
    #  Is this good practice?  No.  But hydra doesn't give a great alternative
    import sys
    if "--help" not in sys.argv and "--hydra-help" not in sys.argv:
        sys.argv += ['hydra/job_logging=disabled']
    main()
