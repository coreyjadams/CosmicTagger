#!/usr/bin/env python
import os,sys,signal
import time

import numpy


#############################

# Add the local folder to the import path:
network_dir = os.path.dirname(os.path.abspath(__file__))
network_dir = os.path.dirname(network_dir)
sys.path.insert(0,network_dir)

import argparse

from src.networks.config import str2bool

class exec(object):

    def __init__(self):

        # This technique is taken from: https://chase-seibert.github.io/blog/2014/03/21/python-multilevel-argparse.html
        parser = argparse.ArgumentParser(
            description='Run Cosmic Tagger application',
            usage='''exec.py <command> [<args>]

The most commonly used commands are:
   train         Train a network, either from scratch or restart
   inference     Run inference with a trained network
   iotest        Run IO testing without training a network
''')
        parser.add_argument('command', help='Subcommand to run')
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print(f'Unrecognized command {args.command}')
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def add_shared_training_arguments(self, parser):

        ##################################################################
        # Parameters to control logging and snapshotting
        ##################################################################
        parser.add_argument('-ci','--checkpoint-iteration',
            type    = int,
            default = 100,
            help    = 'Period (in steps) to store snapshot of weights')

        parser.add_argument('-si','--summary-iteration',
            type    = int,
            default = 1,
            help    = 'Period (in steps) to store summary in tensorboard log')

        parser.add_argument('--no-summary-images',
            type    = str2bool,
            default = False,
            help    = 'Skip summary images to save on memory')

        parser.add_argument('-li','--logging-iteration',
            type    = int,
            default = 1,
            help    = 'Period (in steps) to print values to log')

        parser.add_argument('-cd','--checkpoint-directory',
            type    = str,
            default = None,
            help    = 'Directory to store model snapshots')

        parser.add_argument('--gradient-accumulation',
            type    = int,
            default = 1,
            help    = "Accumulate this many minibatches before updating weigths.")

        ##################################################################
        # Parameters to control the network training
        ##################################################################

        parser.add_argument('-lr','--learning-rate',
            type    = float,
            default = 0.0003,
            help    = 'Initial learning rate')

        parser.add_argument('--optimizer',
            type    = str,
            choices = ['adam', 'rmsprop',],
            default = 'rmsprop',
            help    = 'Optimizer to use')

        parser.add_argument('--loss-balance-scheme',
            type    = str,
            choices = ['none', 'focal', 'even', 'light'],
            default = 'none',
            help    = "Way to compute weights for balancing the loss.")

        parser.add_argument('--weight-decay',
            type    = float,
            default = 0.0,
            help    = "Weight decay strength")

        parser.add_argument('-rw','--regularize-weights',
            type    = float,
            default = 0.00001,
            help    = "Regularization strength for all learned weights")


        ##################################################################
        ### Torch Specific
        ##################################################################

        parser.add_argument('--mixed-precision',
            type    = str2bool,
            default = False,
            help    = "Use mixed precision for training.")

        parser.add_argument('--loss-scale',
            type    = float,
            default = 1.0,
            help    = "Amount to scale the loss function before back prop.")

        ##################################################################
        ### Tensorflow Specific
        ##################################################################

        parser.add_argument('--inter-op-parallelism-threads',
            type    = int,
            default = 4,
            help    = "Passed to tf configproto.")

        parser.add_argument('--intra-op-parallelism-threads',
            type    = int,
            default = 24,
            help    = "Passed to tf configproto.")

        ##################################################################
        # High level network decisions: 2D/3D, Sparse/Dense
        ##################################################################

        parser.add_argument('--conv-mode',
            type    = str,
            default = '2D',
            choices = ['2D','3D'],
            help    = "Only for non-sparse (dense) mode, use 2d or 3d convolutions.")



    def train(self):
        self.parser = argparse.ArgumentParser(
            description     = 'Run Network Training',
            formatter_class = argparse.ArgumentDefaultsHelpFormatter)

        self.add_io_arguments(self.parser)
        self.add_core_configuration(self.parser)
        self.add_shared_training_arguments(self.parser)

        self.add_network_parser(self.parser)

        self.args = self.parser.parse_args(sys.argv[2:])
        self.args.training = True
        self.args.mode = "train"


        self.make_trainer()

        self.trainer.print("Running Training")
        self.trainer.print(self.__str__())

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
        self.parser = argparse.ArgumentParser(
            description     = 'Run IO Testing',
            formatter_class = argparse.ArgumentDefaultsHelpFormatter)
        self.add_io_arguments(self.parser)
        self.add_core_configuration(self.parser)

        # now that we're inside a subcommand, ignore the first
        # TWO argvs, ie the command (exec.py) and the subcommand (iotest)
        self.args = self.parser.parse_args(sys.argv[2:])
        self.args.training = False
        self.args.mode = "iotest"

        self.make_trainer()
        
        self.trainer.print("Running IO Test")
        self.trainer.print(self.__str__())


        self.trainer.initialize(io_only=True)

        # label_stats = numpy.zeros((36,))
        global_start = time.time()
        time.sleep(0.1)
        for i in range(self.args.iterations):
            start = time.time()
            mb = self.trainer.larcv_fetcher.fetch_next_batch("train", force_pop=True)

            end = time.time()

            self.trainer.print(i, ": Time to fetch a minibatch of data: {}".format(end - start))

        self.trainerprint("Total IO Time: ", time.time() - global_start)
    def make_trainer(self):

        self.validate_arguments()

        if self.args.mode == "iotest":
            from src.utils.core import trainercore
            self.trainer = trainercore.trainercore(self.args)
            return

        if self.args.framework == "tensorflow" or self.args.framework == "tf":


            if self.args.distributed:
                from src.utils.tensorflow import distributed_trainer
                self.trainer = distributed_trainer.distributed_trainer(self.args)
            else:
                from src.utils.tensorflow import trainer
                self.trainer = trainer.tf_trainer(self.args)

        elif self.args.framework == "torch":
            if self.args.distributed:
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

        print("Running Inference")
        print(self.__str__())

        self.trainer.initialize()
        self.trainer.batch_process()

    def __str__(self):
        s = "\n\n-- CONFIG --\n"
        for name in iter(sorted(vars(self.args))):
            # if name != name.upper(): continue
            attribute = getattr(self.args,name)
            # if type(attribute) == type(self.parser): continue
            # s += " %s = %r\n" % (name, getattr(self, name))
            substring = ' {message:{fill}{align}{width}}: {attr}\n'.format(
                   message=name,
                   attr = getattr(self.args, name),
                   fill='.',
                   align='<',
                   width=30,
                )
            s += substring
        return s




    def add_core_configuration(self, parser):
        # These are core parameters that are important for all modes:
        parser.add_argument('-i', '--iterations',
            type    = int,
            default = 25000,
            help    = "Number of iterations to process")

        parser.add_argument('-d','--distributed',
            action  = 'store_true',
            default = False,
            help    = "Run with the MPI compatible mode")

        parser.add_argument('-m','--compute-mode',
            type    = str,
            choices = ['CPU','GPU'],
            default = 'GPU',
            help    = "Selection of compute device, CPU or GPU ")

        parser.add_argument('-ld','--log-directory',
            default ="log/",
            help    ="Prefix (directory) for logging information")


        ##################################################################
        # Parameters to control framework options
        ##################################################################

        parser.add_argument('--framework',
            type    = str, choices=['torch','tensorflow', 'tf'],
            default = "tensorflow",
            help    = "Pick to use either torch or tensorflow/tf.")

        return parser

    def add_io_arguments(self, parser):

        # data_directory = "/lus/theta-fs0/projects/datascience/cadams/datasets/SBND/H5/cosmic_tagging/"
        # data_directory = "/Users/corey.adams/data/dlp_larcv3/sbnd_cosmic_samples/cosmic_tagging/"
        data_directory = "/gpfs/jlse-fs0/users/cadams/datasets/cosmic_tagging/"

        # IO PARAMETERS FOR INPUT:
        parser.add_argument('-f','--file',
            type    = str,
            default = data_directory + "cosmic_tagging_train.h5",
            help    = "IO Input File")

        parser.add_argument('--start-index',
            type    = int,
            default = 0,
            help    = "Start index, only used in inference mode")

        parser.add_argument('-mb','--minibatch-size',
            type    = int,
            default = 2,
            help    = "Number of images in the minibatch size")

        # IO PARAMETERS FOR AUX INPUT:
        parser.add_argument('--aux-file',
            type    = str,
            # default = None,
            default = data_directory + "cosmic_tagging_test.h5",
            help    = "IO Aux Input File, or output file in inference mode")


        parser.add_argument('--aux-iteration',
            type    = int,
            default = 10,
            help    = "Iteration to run the aux operations")

        parser.add_argument('--aux-minibatch-size',
            type    = int,
            default = 2,
            help    = "Number of images in the minibatch size")

        parser.add_argument('--synthetic',
            type    = str2bool,
            default = False ,
            help    = "Use synthetic data instead of real data.")

        parser.add_argument('-df','--data-format',
            type    = str,
            choices = ["channels_last", "channels_first"],
            default = "channels_last",
            help    = "Channels format in the tensor shape.")

        parser.add_argument('-ds', '--downsample-images',
            default = 1,
            type    = int,
            help    = 'Dense downsampling of the images.  This is the number of downsamples applied (0 == none, 1 == once ...) ')


        parser.add_argument('--sparse',
            type    = str2bool,
            default = False,
            help    = "Use sparse convolutions instead of dense convolutions.")

        return

    def validate_arguments(self):

        if self.args.framework == "torch":
            # In torch, only option is channels first:
            if self.args.data_format == "channels_last":
                print("Torch requires channels_first, switching automatically")
                self.args.data_format = "channels_first"


if __name__ == '__main__':
    s = exec()
