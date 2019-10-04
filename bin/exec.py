#!/usr/bin/env python
import os,sys,signal
import time

import numpy


# Add the local folder to the import path:
network_dir = os.path.dirname(os.path.abspath(__file__))
network_dir = os.path.dirname(network_dir)
sys.path.insert(0,network_dir)


# import the necessary
from src.utils.core import flags


def main():

    # If you extend the flags class, change this line!
    FLAGS = flags.uresnet()
    FLAGS.parse_args()
    # FLAGS.dump_config()



    if FLAGS.MODE is None:
        raise Exception()

    if FLAGS.DISTRIBUTED:
        if FLAGS.FRAMEWORK == "tf" or FLAGS.FRAMEWORK == "tensorflow":
            from src.utils.tensorflow import distributed_trainer
            model_trainer = distributed_trainer.distributed_trainer()
        elif FLAGS.FRAMEWORK == "torch":
            from src.utils.torch import distributed_trainer
            model_trainer = distributed_trainer.distributed_trainer()
    else:
        if FLAGS.FRAMEWORK == "tf" or FLAGS.FRAMEWORK == "tensorflow":
            from src.utils.tensorflow import trainer
            model_trainer = trainer.tf_trainer()
        elif FLAGS.FRAMEWORK == "torch":
            from src.utils.torch import trainer
            model_trainer = trainer.torch_trainer()

    if FLAGS.MODE == 'train' or FLAGS.MODE == 'inference':

        # On these lines, you would get the network class and pass it to FLAGS
        # which can share it with the trainers.  This lets you configure the network
        # without having to rewrite the training interface each time.
        # It would look like this:

        model_trainer.initialize()
        model_trainer.batch_process()

    if FLAGS.MODE == 'iotest':
        model_trainer.initialize(io_only=True)

        total_start_time = time.time()
        # time.sleep(0.1)
        start = time.time()
        for i in range(FLAGS.ITERATIONS):
            mb = model_trainer.fetch_next_batch()
            end = time.time()

            if not FLAGS.DISTRIBUTED:
                print(i, ": Time to fetch a minibatch of data: {}".format(end - start))
            else:
                if model_trainer._rank == 0:
                    print(i, ": Time to fetch a minibatch of data: {}".format(end - start))
            # time.sleep(0.5)
            start = time.time()

            # model_trainer._larcv_interface.prepare_next('primary')

        total_time = time.time() - total_start_time
        print("Time to read {} batches of {} images each: {}".format(
            FLAGS.ITERATIONS,
            FLAGS.MINIBATCH_SIZE,
            time.time() - total_start_time
            ))

    model_trainer.stop()

if __name__ == '__main__':
    main()
