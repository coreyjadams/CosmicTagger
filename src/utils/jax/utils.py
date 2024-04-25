import jax
import jax.numpy as numpy

import optax

from flax.training import orbax_utils
from orbax import checkpoint

import pathlib

from src.config import Precision


def prepare_input_data(args, default_device_context, minibatch_data):
    
    # with default_device_context():

    # print(minibatch_data["image"].dtype)

    if args.run.precision == Precision.bfloat16:
        minibatch_data["image"] = minibatch_data["image"].astype(numpy.bfloat16)

    labels_dict = {
        "segmentation" : numpy.split(minibatch_data['label'].astype(numpy.int32),
                                        indices_or_sections=3, 
                                        axis=-1),
    }

    if args.network.classification.active or args.network.vertex.active:
        labels_dict.update({"event_label"  : minibatch_data['event_label']})
    if args.network.vertex.active:
        labels_dict.update({"vertex"  : minibatch_data['vertex']})
    shape = labels_dict["segmentation"][0].shape
    labels_dict["segmentation"] = [
        _label.reshape(shape[:-1])
            for _label in labels_dict["segmentation"]
    ]

    return minibatch_data, labels_dict



from src.networks.jax  import LossCalculator, AccuracyCalculator

def create_train_val_steps(args, default_device_context, reduction_op):
    
    # Define the loss function:
    if args.network.classification.active:
        with default_device_context:
            weight = numpy.asarray([0.16, 0.1666, 0.16666, 0.5])
            if args.run.precision == Precision.bfloat16:
                weight = weight.to(numpy.bfloat16)
            loss_calculator = LossCalculator(args, weight=weight)
    else:
        loss_calculator = LossCalculator(args)

    compute_metrics = AccuracyCalculator(args)


    def create_train_step():


        def train_step(batch, state):

            # Pull off the apply function:
            apply_fn = state.apply_fn
            
            def forward_to_loss(parameters, network_inputs, batch_labels):

                # True is training
                logits = apply_fn(parameters, network_inputs, True)
                loss, metrics = loss_calculator(batch_labels, logits)
                # Returning this as loss, aux:
                return loss, (logits, metrics)


            batch, labels = prepare_input_data(args, default_device_context, batch)

            # Go forward:
            grad_fn = jax.value_and_grad(forward_to_loss, has_aux=True)

            (loss, (logits, loss_metrics)), grads = grad_fn(state.params, batch["image"], labels)


            acc_metrics = compute_metrics(logits, labels)

            # This is a no-op in single-rank mode:
            grads = reduction_op(grads)

            state = state.apply_gradients(grads=grads)
            # This could be an issues, it's not a pure function:
            acc_metrics.update(loss_metrics)

            return state, acc_metrics
    

        return train_step
    
    def create_val_step():
    
        def val_step(batch, state):
        
            # Pull off the apply function:
            apply_fn = state.apply_fn
            
            batch, labels = prepare_input_data(args, default_device_context, batch)
            
            # False is training=False
            logits = apply_fn(state.params, batch["image"], False)
            loss, metrics = loss_calculator(labels, logits)

            acc_metrics = compute_metrics(logits, labels)

            # This could be an issues, it's not a pure function:
            acc_metrics.update(metrics)

            return acc_metrics




        return val_step
    
    

    return create_train_step(), create_val_step()


def init_checkpointer(save_path, restore_path=None, should_do_io=False):

    if restore_path is None:
        restore_path = save_path

    restore_ckpt_path = pathlib.Path(restore_path) / pathlib.Path("checkpoint") / pathlib.Path("model")
    save_ckpt_path    = pathlib.Path(save_path)    / pathlib.Path("checkpoint") / pathlib.Path("model")
    
    restore_checkpointer = checkpoint.PyTreeCheckpointer()
    save_checkpointer    = checkpoint.PyTreeCheckpointer()

    options = checkpoint.CheckpointManagerOptions(max_to_keep=5, create=True)
    if should_do_io:
        checkpoint_manager = checkpoint.CheckpointManager(
            save_ckpt_path.resolve(),
            save_checkpointer,
            options
        )
        restore_manager = checkpoint.CheckpointManager(
            restore_ckpt_path.resolve(),
            restore_checkpointer,
            options
        )
    

        def save_weights(train_state):

            ckpt = {
                'model' : train_state.params,
                'opt'   : train_state.opt_state,
            }
            save_args = orbax_utils.save_args_from_target(ckpt)

            checkpoint_manager.save(train_state.step, ckpt, save_kwargs={'save_args': save_args})

            return


        def restore_weights():

            global_step = restore_manager.latest_step()

            checkpoint = restore_manager.restore(global_step)
            restored_state = checkpoint['model']
            # restored_model = target_type(checkpoint['model'])
            restored_opt   = checkpoint['opt']

            return restored_state, restored_opt, global_step


        return save_weights, restore_weights
    
    else:

        return lambda * args, **kwargs : None, lambda * args, **kwargs : None


def init_optax_lr_schedule(original_schedule):
    '''
    Convert a python schedule into an optax schedule to hack jax together
    '''

    from ..core import learning_rate_scheduler


    if isinstance(original_schedule, learning_rate_scheduler.FlatSchedule):
        return optax.constant_schedule(original_schedule(1))
    # elif isinstance
    else:
        raise Exception(f"Couldn't convert {type(original_schedule)} to optax")