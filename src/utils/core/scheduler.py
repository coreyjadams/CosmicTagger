import numpy

'''

The philosophy of this learning rate scheduler is:

- Particularly with distributed learning, varying the learning rate is important
- The interface for the learning rate variations are clunky in both TF and torch
- I can wrap their interfaces around a more flexible class suited for this, and
  use a custom class to handle intracies in a way that is identical between torch
  and tf.
- we use the initial_step to allow restarting

[description]
'''

class learning_rate_scheduler(object):
    '''
    Generalized interface for computing a current learning rate based on
    a schedule.  The constructor takes the arguments for total_run_length,
    iterations per epoch,
    '''

    def __init__(self, *,
        total_run_iterations = None,
        total_run_epochs     = None,
        batch_size           = None,
        dataset_size         = None,
        learning_schedule    = None,
        initial_step         = 0,
    ):



        if total_run_epochs is not None and total_run_iterations is not None:
            raise Exception("Either set total epochs or total iterations, not both")

        schedule_options =  ["constant", "warm_up", "cyclic", "linear_ramp"]

        if learning_schedule is not in schedule_options:
            raise Exception("Learning rate schedule must be in the following: ", schedule_options)


        self.current_step = initial_step

        # We need to know the total run length either in terms of iterations or epochs
        # total_epochs = total_run_iterations * (dataset_size / batch_size)


    def get_current_learning_rate(self, *,
        current_iteration = None,
        current_epoch     = None,
    ):


        if current_epoch is not None and current_iteration is not None:
            raise Exception("Pass either current iteration or current epoch")
