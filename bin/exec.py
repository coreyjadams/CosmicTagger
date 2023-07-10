#!/usr/bin/env python
import os,sys,signal
import time
import pathlib
# import logging
# from logging import handlers

import numpy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    import tensorflow as tf
    tf.get_logger().setLevel('INFO')
except:
    pass

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

from src.config import Config, RunUnit, ComputeMode, DataFormatKind

from src.config.mode import ModeKind

from src.utils.io import create_larcv_dataset

# Custom logger:
from src.utils import logging

import atexit

class exec(object):

    def __init__(self, config):

        self.args = config

        rank = self.init_mpi()


        # Add to the output dir:
        self.args.output_dir += f"/{self.args.framework.name}/"
        self.args.output_dir += f"/{self.args.network.name}/"
        self.args.output_dir += f"/{self.args.run.id}/"
        self.args.output_dir += f"/ds{self.args.data.downsample}/"

        # Create the output directory if needed:
        if rank == 0:
            outpath = pathlib.Path(self.args.output_dir)
            outpath.mkdir(exist_ok=True, parents=True)

        self.configure_logger(rank)

        self.validate_arguments()

        # Print the command line args to the log file:
        logger = logging.getLogger("CosmicTagger")
        logger.info("Dumping launch arguments.")
        logger.info(sys.argv)
        logger.info(self.__str__())

        logger.info("Configuring Datasets.")
        self.datasets = self.configure_datasets()
        logger.info("Data pipeline ready.")


    def run(self):
        if self.args.mode.name == ModeKind.iotest:
            self.iotest()
        else:
            self.batch_process()

    def exit(self):
        if hasattr(self, "trainer"):
            self.trainer.exit()

    def init_mpi(self):
        if not self.args.run.distributed:
            return 0
        else:
            from src.utils.core import mpi_init_and_local_rank
            local_rank = mpi_init_and_local_rank(set_env=True, verbose=False)

            return int(os.environ["RANK"])

    def configure_lr_schedule(self, epoch_length, max_epochs):


        if self.args.mode.optimizer.lr_schedule.name == "one_cycle":
            from src.utils.core import OneCycle
            lr_schedule = OneCycle(self.args.mode.optimizer.lr_schedule)
        elif self.args.mode.optimizer.lr_schedule.name == "standard":
            from src.utils.core import WarmupFlatDecay
            schedule_args = self.args.mode.optimizer.lr_schedule
            lr_schedule = WarmupFlatDecay(
                peak_learning_rate = schedule_args.peak_learning_rate,
                decay_floor  = schedule_args.decay_floor,
                epoch_length = epoch_length,
                decay_epochs = schedule_args.decay_epochs,
                total_epochs = max_epochs
            )

        return lr_schedule

    def configure_datasets(self):
        """
        This function creates the non-framework iterable datasets used in this app.

        They get converted to framework specific tools, if needed, in the
        framework specific code.
        """

        # Check if we need vertex or eventID info:
        event_id = False
        if hasattr(self.args.network, "classification"):
            if self.args.network.classification.active:
                event_id = True

        vertex_depth = None
        if hasattr(self.args.network, "vertex"):
            if self.args.network.vertex.active:
                # Convert the vertex depth to measure from the top down here:
                vertex_depth = self.args.network.depth - self.args.network.vertex.depth
                event_id = True

        # Manually override - here - the data format in some cases
        import copy
        
        data_args = copy.copy(self.args.data)
        if self.args.run.compute_mode == ComputeMode.XPU:
            if self.args.framework.name == "torch":
                data_args.data_format = DataFormatKind.channels_first

        if self.args.data.synthetic:
            if self.args.mode.name == ModeKind.train:
                name = "train"
            else:
                name = "test"

            datasets = {
                name : create_larcv_dataset(
                    data_args    = data_args,
                    batch_size   = self.args.run.minibatch_size,
                    input_file   = None,
                    name         = name,
                    distributed  = False,
                    event_id     = event_id,
                    vertex_depth = vertex_depth,
                    sparse       = False)
            }
        else:
            datasets = {
                name : create_larcv_dataset(
                    data_args    = data_args,
                    batch_size   = self.args.run.minibatch_size,
                    input_file   = getattr(self.args.data.paths, name),
                    name         = name,
                    distributed  = self.args.run.distributed,
                    event_id     = event_id,
                    vertex_depth = vertex_depth,
                    sparse       = self.args.framework.sparse)
                for name in self.args.data.paths.active
            }

        return datasets


    def configure_logger(self, rank):

        logger = logging.getLogger("CosmicTagger")
        if rank == 0:
            logger.setFile(self.args.output_dir + "/process.log")
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(999)

        # logger = logging.getLogger("CosmicTagger")
        # logger.propogate=False
        # # Create a handler for STDOUT, but only on the root rank.
        # # If not distributed, we still get 0 passed in here.
        # if rank == 0:
        #     stream_handler = logging.StreamHandler(sys.stdout)
        #     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        #     stream_handler.setFormatter(formatter)
        #     handler = handlers.MemoryHandler(capacity = 0, target=stream_handler)
        #     handler.setLevel(logging.INFO)
        #     logger.addHandler(handler)
        #
        #     # Add a file handler too:
        #     log_file = self.args.output_dir + "/process.log"
        #     file_handler = logging.FileHandler(log_file)
        #     file_handler.setFormatter(formatter)
        #     file_handler = handlers.MemoryHandler(capacity=10, target=file_handler)
        #     file_handler.setLevel(logging.INFO)
        #     logger.addHandler(file_handler)
        #
        #     logger.setLevel(logging.INFO)
        # else:
        #     # in this case, MPI is available but it's not rank 0
        #     # create a null handler
        #     handler = logging.NullHandler()
        #     logger.addHandler(handler)
        #     logger.setLevel(logging.INFO)
        #     # logging.getLogger("CosmicTagger").setLevel(logging.ERROR)


    def batch_process(self):

        logger = logging.getLogger("CosmicTagger")

        logger.info(f"Running in mode: {self.args.mode.name.name}")

        self.make_trainer()

        if self.args.framework.name == "lightning":
            from src.utils.torch.lightning import train
            train(self.args, self.trainer, self.datasets, self.max_epochs, self.max_steps)
        else:
            self.trainer.initialize(self.datasets)
            self.trainer.batch_process(self.datasets, self.max_epochs, self.max_steps)


    def iotest(self):

        logger = logging.getLogger("CosmicTagger")

        logger.info("Running IO Test")


        # self.trainer.initialize(io_only=True)

        if self.args.run.distributed:
            from mpi4py import MPI
            rank = MPI.COMM_WORLD.Get_rank()
        else:
            rank = 0




        for key, dataset in self.datasets.items():
            logger.info(f"Reading dataset {key}")
            global_start = time.time()
            total_reads = 0


            # Determine the stopping point:
            if self.args.run.run_units == RunUnit.epoch:
                break_i = self.args.run.run_length * len(dataset)
            else:
                break_i = self.args.run.run_length

            start = time.time()
            for i, minibatch in enumerate(dataset):
                print(minibatch['image'].shape)
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

    def log_keys(self):

        log_keys = ['Average/Non_Bkg_Accuracy', 'Average/mIoU']
        if hasattr(self.args.network, "classification"):
            if self.args.network.classification.active:
                log_keys += ['Average/EventLabel',]
        if hasattr(self.args.network, "vertex"):
            if self.args.network.vertex.active:
                log_keys += ['Average/VertexDetection',]
        if self.args.mode.name == ModeKind.train:
            log_keys.append("loss/total")

        return log_keys

    def hparams_keys(self):

        # Copy these:
        hparams_keys = [ lk for lk in  self.log_keys()]
        # Add to it
        hparams_keys += ["Average/Neutrino_IoU"]
        hparams_keys += ["Average/Cosmic_IoU"]
        hparams_keys += ["Average/Total_Accuracy"]
        hparams_keys += ["loss/segmentation"]
        if hasattr(self.args.network, "classification"):
            if self.args.network.classification.active:
                hparams_keys += ['loss/event_label',]
        if hasattr(self.args.network, "vertex"):
            if self.args.network.vertex.active:
                hparams_keys += ['Average/VertexResolution',]
                hparams_keys += ['loss/vertex/detection',]
                hparams_keys += ['loss/vertex/localization',]

        return hparams_keys


    def set_run_length_info(self, dataset_length):
        """
        Compute the total number of epochs, and length per epoch

        Sets state variables self.max_epochs and self.epoch_length
        """


        # Need to configure epoch length and number of epochs for the scheduler and trainer:
        from src.config import RunUnit
        if self.args.run.run_units == RunUnit.epoch:
            self.max_epochs   = self.args.run.run_length
            self.epoch_length = int(dataset_length / self.args.run.minibatch_size)
            self.max_steps    = self.max_epochs * self.epoch_length
        elif self.args.run.run_units == RunUnit.iteration:
            # Max steps is easy:
            self.max_steps    = self.args.run.run_length
            
            self.epoch_length = dataset_length
            # This is totally arbitrary but meant to ensure there are enough
            # epochs in the lr scheduler
            # It's not recommended to use this mode unless benchmarking.
            self.max_epochs = None

        return

    def make_trainer(self):

        # Set the random seed for numpy, which controls the order of the 
        # data loading:
        data_seed = self.args.data.seed
        if data_seed < 0:
            data_seed = int(time.time())
        numpy.random.seed(data_seed)


        framework_seed = self.args.framework.seed
        if framework_seed == 0:
            framework_seed = int(time.time())


        if 'environment_variables' in self.args.framework:
            for env in self.args.framework.environment_variables.keys():
                os.environ[env] = self.args.framework.environment_variables[env]

        dataset_length = max([len(ds) for ds in self.datasets.values()])

        self.set_run_length_info(dataset_length)


        if self.args.mode.name == ModeKind.train:
            lr_schedule = self.configure_lr_schedule(self.epoch_length, self.max_epochs)
        else:
            lr_schedule = None

        if self.args.framework.name == "tensorflow":

            import logging
            logging.getLogger('tensorflow').setLevel(logging.FATAL)


            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

            # Import tensorflow and see what the version is.
            import tensorflow as tf
            if self.args.framework.seed != 0:
                tf.config.experimental.enable_op_determinism()
            # tf.keras.utils.set_random_seed(framework_seed)
            tf.random.set_seed(framework_seed)
            import random; random.seed(framework_seed)

            if self.args.run.distributed:
                from src.utils.tensorflow2 import distributed_trainer
                self.trainer = distributed_trainer.distributed_trainer(self.args,
                    self.datasets,
                    lr_schedule,
                    log_keys     = self.log_keys(),
                    hparams_keys = self.hparams_keys()
                )
            else:
                from src.utils.tensorflow2 import trainer
                self.trainer = trainer.tf_trainer(self.args,
                    self.datasets,
                    lr_schedule,
                    log_keys     = self.log_keys(),
                    hparams_keys = self.hparams_keys()
                )

        elif self.args.framework.name == "torch":

            # Import tensorflow and see what the version is.
            if self.args.framework.seed != 0:
                import torch
                torch.manual_seed(self.args.framework.see)
                torch.use_deterministic_algorithms(True)
                
                # Seed python too:
                import random; random.seed(framework_seed)

            if self.args.run.distributed:
                from src.utils.torch import distributed_trainer
                self.trainer = distributed_trainer.distributed_trainer(
                    self.args,
                    self.datasets,
                    lr_schedule,
                    log_keys     = self.log_keys(),
                    hparams_keys = self.hparams_keys()
                    )
            else:
                from src.utils.torch import trainer
                self.trainer = trainer.torch_trainer(
                    self.args,
                    self.datasets,
                    lr_schedule,
                    log_keys     = self.log_keys(),
                    hparams_keys = self.hparams_keys()
                    )

        elif self.args.framework.name == "lightning":
            from src.utils.torch.lightning import create_lightning_module
            self.trainer = create_lightning_module(
                self.args,
                self.datasets,
                lr_schedule,
                log_keys     = self.log_keys(),
                hparams_keys = self.hparams_keys())


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

        logger = logging.getLogger("CosmicTagger")

        if self.args.framework.name == "torch" or self.args.framework.name == "lightning":
            # In torch, only option is channels first:
            if self.args.data.data_format == DataFormatKind.channels_last:
                if self.args.run.compute_mode == ComputeMode.CUDA:
                    logger.warning("CUDA Torch requires channels_first, switching automatically")
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
