import os
import sys
import time
from collections import OrderedDict

import numpy

<<<<<<< HEAD
from .trainercore import trainercore

import tensorflow as tf

for i, p in enumerate(sys.path):
    if ".local" in p:
        sys.path.pop(i)

import horovod.tensorflow as hvd
hvd.init()



os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())

from . import flags
FLAGS = flags.FLAGS()

if not FLAGS.SYNTHETIC:
    from larcv.distributed_queue_interface import queue_interface

=======
import torch
import horovod.torch as hvd
hvd.init()

from larcv.distributed_larcv_interface import larcv_interface

from mpi4py import MPI
comm = MPI.COMM_WORLD

from . import flags
# from . import data_transforms
FLAGS = flags.FLAGS()

from .trainercore import trainercore

import tensorboardX


def lambda_warmup(epoch):
    # Constant terms:
    flat_warmup = 50
    linear_warmup = 50
    full = 500
    size=hvd.size()
    target = numpy.sqrt(size)
    # Perform 500 warmup steps, gradually ramping the rate:
    if epoch <= flat_warmup:
        return 1.0
    elif epoch < flat_warmup + linear_warmup:
        return 1.0 + (target - 1) * (epoch - flat_warmup) / linear_warmup 
    elif epoch <= flat_warmup + linear_warmup + full:
        return target
    else:
        return target * numpy.exp(-0.001*(epoch-(full+linear_warmup+flat_warmup)))

# def lr_increase(step):

#     # This function actually ignores the input and uses the global step variable
#     # This allows it to get the learning rate correct after restore.

#     # For this problem, the dataset size is 1e5.
#     # So the epoch can be calculated easily:
#     # epoch = (step * FLAGS.MINIBATCH_SIZE) / (1e5)

#     base_lr   = FLAGS.LEARNING_RATE
#     step_size = 5.0

#     return 1.0 + step*step_size

# peak_lr = 0.5
# cycle_len = 0.8

# def one_cycle_clr(step):
    
#     peak = peak_lr / FLAGS.LEARNING_RATE
    
#     cycle_steps  = int(FLAGS.ITERATIONS*cycle_len)
#     end_steps = FLAGS.ITERATIONS - cycle_steps
#     # Which cycle are we in?

#     cycle = int(step / cycle_steps)
#     intra_step = 1.0 * (step % cycle_steps)

#     base_multiplier = 1.0

#     if cycle < 1:
# #         base_multiplier *= 0.5
        
#         if intra_step > cycle_steps*0.5:
#             intra_step = cycle_steps - intra_step

#         value = intra_step * (peak) /(0.5*cycle_steps) 

#     else:
#         value = (intra_step / end_steps)*-1.0

#     # print("Step: {}, Cycle: {}, base {}, intra_step {}, value: {}, total_scale: {}".format(
#     #     step, cycle, base_multiplier, intra_step, value, base_multiplier + value)
#     # )

#     return base_multiplier + value
>>>>>>> torch


class distributed_trainer(trainercore):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self):
        # Rely on the base class for most standard parameters, only
        # search for parameters relevant for distributed computing here

        # Put the IO rank as the last rank in the COMM, since rank 0 does tf saves
<<<<<<< HEAD
        # root_rank = hvd.size() - 1 

        self._larcv_interface = queue_interface()
        self._iteration       = 0
        self._rank            = hvd.rank()

        self._val_writer      = None
=======
        root_rank = hvd.size() - 1 

        if FLAGS.COMPUTE_MODE == "GPU":
            os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())
            

        self._larcv_interface = larcv_interface(root=root_rank)
        self._iteration       = 0
        self._rank            = hvd.rank()
        self._cleanup         = []

>>>>>>> torch

        # Make sure that 'LEARNING_RATE' and 'TRAINING'
        # are in net network parameters:

<<<<<<< HEAD
        self._metrics = OrderedDict()

        self._cleanup         = []


    def set_compute_parameters(self):

        self._config = tf.ConfigProto()


        if FLAGS.COMPUTE_MODE == "CPU":
            self._config.inter_op_parallelism_threads = FLAGS.INTER_OP_PARALLELISM_THREADS
            self._config.intra_op_parallelism_threads = FLAGS.INTRA_OP_PARALLELISM_THREADS
        if FLAGS.COMPUTE_MODE == "GPU":
            self._config.gpu_options.allow_growth = True
            # This is managed with the env variable at the top:
            # self._config.gpu_options.visible_device_list = str(hvd.local_rank())


    def initialize(self, io_only=False):



        tf.logging.info("HVD rank: {}".format(hvd.rank()))
        self.set_compute_parameters()

        self._initialize_io(color=self._rank)

        if io_only:
            return


        if hvd.rank() == 0:
            print(FLAGS.dump_config())



        graph = tf.get_default_graph()

        net_time = self.init_network()
        if hvd.rank() == 0:
            sys.stdout.write("Done constructing network. ({0:.2}s)\n".format(net_time))

        if hvd.rank() == 0:
            self.print_network_info()



        self.init_optimizer()

        self.init_saver()

        # Take all of the metrics and turn them into summaries:
        for key in self._metrics:
            tf.summary.scalar(key, self._metrics[key])

        self._summary_basic = tf.summary.merge_all()
        self._summary_images = self._create_summary_images(self._input['label'], self._output['prediction'])



        # Add the graph to the log file:
        if hvd.rank() == 0:
            self._main_writer.add_graph(graph)


        # Create a session:
        self._sess = tf.Session(config = self._config)

        # Try to restore a model?
        restored = self.restore_model()

        if hvd.rank() == 0:
            if not restored:
                self._sess.run(tf.global_variables_initializer())
        else:
            # Run the initializer on other ranks, else the bcast op won't work
            self._sess.run(tf.global_variables_initializer())
        # Rank 0 has either restored, or has initialized.  Broadcast it:
        g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)



        bcast = hvd.broadcast_variables(g_vars, 0)


        # print(bcast)
        self._sess.run(bcast)




        with tf.variable_scope("hvd"):



            # # All reduce metrics:
            for key in self._metrics:
                self._metrics[key] = hvd.allreduce(self._metrics[key])

            # # Create a set of summary objects:
            # for key in self._metrics:
            #     tf.summary.scalar(key, self._metrics[key])

            # In the distributed case, we may want a learning rate behavior:
            self._lr = self.generate_learning_rate(FLAGS.LEARNING_RATE, self._global_step)



            # Wrap the optimizer it in horovod:
            self._opt = hvd.DistributedOptimizer(self._opt)

            self._train_op = self._opt.minimize(self._loss, self._global_step)

            # Here, we have to replace all of the metrics with the allreduced version:

            # Take all of the metrics and turn them into summaries:




    def restore_model(self):
        # Restore model has to restore on one rank and broadcast to other ranks
        if hvd.rank() == 0:
            restored = trainercore.restore_model(self)
            return restored
        else:
            return None        



    def generate_learning_rate(self, 
        base_learning_rate, 
        global_step,
        warmup_steps = 100,
        decay_after_step=4500):

        ''' Compute the peak learning rate, the start point, and such
        '''

        # Learning rate scales linearly from a very low value to the base learning rate *sqrt(N)
        # for the duration of the warm up steps.

        # After some number of steps, the learning rate undergoes a decay by 10

        # For the calculations, we need to set some constants:

        core_learning_rate = tf.constant(base_learning_rate*numpy.sqrt(hvd.size()), dtype=tf.float32)
        initial_learning_rate = tf.constant(0.0001, dtype=tf.float32)
        warmup_steps = tf.constant(warmup_steps, dtype = tf.float32)


        # So, there are 3 phases: warm up, steady state, decay


        # First, have to decide what state we are in.

        scaled_learning_rate = initial_learning_rate +  core_learning_rate * tf.cast(global_step, tf.float32) / warmup_steps

        # Warm up phase:
        this_learning_rate = tf.math.minimum(scaled_learning_rate, core_learning_rate)


        # # Cool down phase:
        # this_learning_rate = tf.cond( global_step > decay_after_step,
        #     lambda: 0.1* core_learning_rate,
        #     lambda: this_learning_rate
        #     )

        lr_summary = tf.summary.scalar("LearningRate", this_learning_rate)

        # # Need to add this directly to the merged summary op:
        # self._summary_basic = tf.summary.merge([lr_summary, self._summary_basic])

        return this_learning_rate


    # def metrics(self, metrics):
    #     # Here, we can allreduce the metrics.

    #     if hvd.rank() == 0:
    #         print metrics

    #     for key in metrics:
    #         metrics[key] = hvd_keras.allreduce(metrics[key])

    #     if hvd.rank() == 0:
    #         print metrics


    def save_model(self, gs):
        if hvd.rank() != 0:
            return
        else:
            trainercore.save_model(self, gs)

    def write_summaries(self, writer, summary, global_step):
        if hvd.rank() != 0:
            return
        else:
            trainercore.write_summaries(self, writer, summary, global_step)

    def log(self, metrics, kind, step):
        if hvd.rank() != 0:
            return
        else:
            trainercore.log(self, metrics, kind, step)
   
    def batch_process(self):

        if hvd.rank() == 0:
            trainercore.batch_process(self, verbose=True)
        else:
            trainercore.batch_process(self, verbose=False)

=======

    def __del__(self):
        if hvd.rank() == 0:
            trainercore.__del__(self)
            

    def save_model(self):

        if hvd.rank() == 0:
            trainercore.save_model(self)
            



    def init_optimizer(self):

        # This takes the base optimizer (self._opt) and replaces
        # it with a distributed version

        trainercore.init_optimizer(self)

        # Wrap the optimizer in a learning rate controller to ensure warmup and 
        # decayed rate at the end.

        # Important: this lambda takes 'epoch' as an argument but it's actually
        # operating on the 'step' parameter.


        self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self._opt, lambda_warmup, last_epoch=-1)


        self._opt = hvd.DistributedOptimizer(self._opt, named_parameters=self._net.named_parameters())




    def init_saver(self):
        if hvd.rank() == 0:
            trainercore.init_saver(self)
        else:
            self._saver = None
            self._aux_saver = None


    def initialize(self, io_only = False):

        print("HVD rank: {}".format(hvd.rank()))

        if self._rank == 0:
            FLAGS.dump_config()
            

        self._initialize_io()


        if io_only:
            return

        self.init_network()


        if FLAGS.TRAINING: 
            self._net.train(True)



        if hvd.rank() == 0:
            n_trainable_parameters = 0
            for var in self._net.parameters():
                n_trainable_parameters += numpy.prod(var.shape)
            print("Total number of trainable parameters in this network: {}".format(n_trainable_parameters))

        self.init_optimizer()

        self.init_saver()
        # hooks = self.get_standard_hooks()


        # Here, either restore the weights of the network or initialize it:
        self._global_step = 0
        # Restore the state from the root rank:
        if hvd.rank() == 0:
            state = self.restore_model()
        else:
            state = None

        if state is not None and hvd.rank() == 0:
            self.load_state(state)

        if FLAGS.MODEL_HALF_PRECISION:
            self._net.half()


        # Broadcast the state of the model:
        hvd.broadcast_parameters(self._net.state_dict(), root_rank = 0)


        if FLAGS.COMPUTE_MODE == "CPU":
            pass
        if FLAGS.COMPUTE_MODE == "GPU":
            self._net.cuda()

        # comm.bcast(self._global_step, root = 0)
        # hvd.broadcast_optimizer_state(self._opt, root_rank = 0)



    def summary(self, metrics, saver=""):
        if hvd.rank() == 0:
            trainercore.summary(self, metrics, saver)
        return
        
    def summary_images(self, logits_image, labels_image, saver=""):
        if hvd.rank() == 0:
            trainercore.summary_images(self, logits_image, labels_image, saver)
        return

    def _compute_metrics(self, logits, minibatch_data, loss):
        # This function calls the parent function which computes local metrics.
        # Then, it performs an all reduce on all metrics:
        metrics = trainercore._compute_metrics(self, logits, minibatch_data, loss)


        for key in metrics:
            # print("All reducing ", key)
            metrics[key] = hvd.allreduce(metrics[key], name = key)

        return metrics

    def on_step_end(self):
        self._lr_scheduler.step()
        pass


    # def to_torch(self, minibatch_data):

    #     # This function wraps the to-torch function but for a gpu forces
    #     # the right device
    #     if FLAGS.COMPUTE_MODE == 'GPU':
    #         device = torch.device('cuda')
    #         # device = torch.device('cuda:{}'.format(hvd.local_rank()))
    #     else:
    #         device = None
    #     minibatch_data = trainercore.to_torch(self, minibatch_data, device)

    #     return minibatch_data

    def log(self, metrics, saver=""):
        if hvd.rank() == 0:
            trainercore.log(self, metrics, saver)
>>>>>>> torch
