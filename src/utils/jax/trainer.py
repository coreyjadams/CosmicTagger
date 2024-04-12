import os

from src.utils import logging

import jax
from jax import numpy
from jax import random
import pandas as pd
import pathlib

import flax
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.training import orbax_utils

import optax
from orbax import checkpoint

from src.utils.core.trainercore import trainercore
from src.networks.jax           import LossCalculator, AccuracyCalculator
# from src.networks.jax           import create_vertex_meta, predict_vertex

import contextlib
@contextlib.contextmanager
def dummycontext():
    yield None

from tensorboardX import SummaryWriter

import datetime
from src.config import ComputeMode, Precision, ConvMode, ModeKind, DataFormatKind, RunUnit



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




def create_train_val_steps(args, default_device_context):
    
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

    print(type(original_schedule))

    if isinstance(original_schedule, learning_rate_scheduler.FlatSchedule):
        return optax.constant_schedule(original_schedule(1))
    # elif isinstance
    else:
        raise Exception(f"Couldn't convert {type(original_schedule)} to optax")

class jax_trainer(trainercore):
    '''
    The JAX Traininer is meant to be relatively light weight.  It uses functional programming
    for many cases and closures to initialize things properly with user parameters.

    The trainer does maintain a flax trainstate, and access to the datasets 
    and it performs the loops.

    '''
    def __init__(self, args, datasets, lr_schedule, log_keys, hparams_keys):
        trainercore.__init__(self, args)

        # self.datasets = datasets
        # Throw out the python lr schedule because we can't jit it:
        self.lr_calculator = init_optax_lr_schedule(lr_schedule)
        # self.lr_calculator = lr_schedule



        self.log_keys      = log_keys
        self.hparams_keys  = hparams_keys

        # trainercore.__init__(self, args)
        self.local_df = []

        # Take the first dataset:
        example_ds = next(iter(datasets.values()))

        # if self.args.network.vertex.active:
        #     self.vertex_meta = create_vertex_meta(args, example_ds.image_meta, example_ds.image_size())

        self.latest_metrics = {}

        # Hold a set of pure functions that offload the main computational parts:
        self.function_lookup = {}


    def init_network(self, image_size, image_meta):
        from src.config import ConvMode

        activation = getattr(nn, self.args.network.activation)


        if self.args.network.conv_mode == ConvMode.conv_2D and not self.args.framework.sparse:
            from src.networks.jax.uresnet2D import UResNet
            self.net = UResNet(self.args.network, image_size, activation)

        key = random.PRNGKey(self.args.framework.seed)

        # Initialize the network:
        params = self.net.init(key, numpy.zeros(image_size + [3,]) )

        # Use a partial to remote the batch norm control flow:

        # VMAP and JIT the model:
        model_fn = jax.vmap(self.net.apply, in_axes=[None, 0, None])

        return params, model_fn

    def initialize(self, datasets):

        example_ds = next(iter(datasets.values()))

        with self.default_device_context():
            params, apply_fn = self.init_network(
                example_ds.image_size(), 
                example_ds.image_meta
            )

            # If using half precision on the model, convert it now:
            if self.args.run.precision == Precision.bfloat16:
                params = jax.tree_util.tree_map(
                    lambda x : x.astype(numpy.bfloat16),
                    params
                )


            self.print_network_info(params)

            if self.is_training():
                opt = self.init_optimizer()
                opt_state = opt.init(params)
                self.train_state = TrainState(
                    step      = 0,
                    apply_fn  = apply_fn,
                    params    = params,
                    tx        = opt,
                    opt_state = opt_state,
                )

            # Initialize savers:
            save_dir = self.args.output_dir
            if not self.args.run.distributed or self.rank == 0:
                self.savers = {
                    ds_name : SummaryWriter(save_dir + f"/{ds_name}/")
                    for ds_name in datasets.keys()
                }
            else:
                self.savers = {ds_name : None for ds_name in datasets.keys()}


            self._global_step = 0

            restore_loc = self.args.mode.weights_location
            if restore_loc == "": restore_loc = None

            # Spin up the save and restore functions:
            self.save_fn, self.restore_fn = init_checkpointer(
                save_dir, 
                restore_path=restore_loc, 
                should_do_io=True
            )

            self.restore_model()

            if self.is_training():

                temp_train_step, temp_val_step = create_train_val_steps(
                    self.args,
                    self.default_device_context(),
                )

                self.function_lookup["train_step"] = jax.jit(temp_train_step)
                self.function_lookup["val_step"]   = jax.jit(temp_val_step)

                # self.function_lookup["train_step"] = temp_train_step
                # self.function_lookup["val_step"]   = temp_val_step


                # And here, create a function that computes the loss and gradients:
                # loss_fn = lambda p, mb : self.loss_calculator(*self.forward_pass(p, mb))

            self.acc_calc = AccuracyCalculator(self.args)

            # For half precision, we disable gradient accumulation.  This is to allow
            # dynamic loss scaling
            if self.args.run.precision == Precision.mixed:
                if self.is_training() and  self.args.mode.optimizer.gradient_accumulation > 1:
                    raise Exception("Can not accumulate gradients in half precision.")


            if self.args.mode.name == ModeKind.inference:
                self.inference_metrics = {}
                self.inference_metrics['n'] = 0


        # Now, we create a training and inference function that takes 


    def print_network_info(self, params, verbose=False):
        logger = logging.getLogger("CosmicTagger")

        

        if verbose:
        
            def resolve_keys(key_list):
                return ".".join([k.key for k in key_list ])
        
            jax.tree_util.tree_map_with_path(
                lambda key, val : logger.info(f"{resolve_keys(key)}: {val.shape}"),
                params
            )


        logger.info("Total number of trainable parameters in this network: {}".format(self.n_parameters(params)))


    def n_parameters(self, params):
        n_trainable_parameters = 0


        for leaf in jax.tree_util.tree_leaves(params):
            n_trainable_parameters += numpy.prod(numpy.array(leaf.shape))

        return n_trainable_parameters

    def restore_model(self):
        logger = logging.getLogger("CosmicTagger")
        return
        # from typing import TypeVar, Any
        # from jax import tree_util
        # TX = TypeVar("TX", bound=optax.OptState)
        # # From https://github.com/google-deepmind/optax/discussions/180        
        # def restore_optimizer_state(opt_state: TX, restored: Any) -> TX:
        #     """Restore optimizer state from loaded checkpoint (or .msgpack file)."""
        #     return tree_util.tree_unflatten(
        #         tree_util.tree_structure(opt_state), tree_util.tree_leaves(restored)
        #     )


        try:
            restored_state, restored_opt, global_step = self.restore_fn()
            restored_opt = restore_optimizer_state(self.train_state.tx, restored_opt)
            # print(self.train_state.opt_state.keys())
            # print(restored_state["opt_state"].keys())
            self.training_state.replace(
                    params    = restored_state,
                    opt_state = restored_opt,
                    step      = global_step,

            )
            self._global_step = global_step

            # if restored_state is not None:
            #     self.train_state = TrainState(
            #         apply_fn  = self.train_state.apply_fn,
            #         tx        = self.train_state.tx,
            #     )
                
        except FileNotFoundError:
            logger.info("Could not restore model because the weights do not exist.")
        finally:
            logger.info("Could not restore model so training from stratch.")
        

    def init_optimizer(self):

        from src.config import OptimizerKind


        lr_function = lambda x : self.lr_calculator(x)

        if self.args.mode.optimizer.name == OptimizerKind.rmsprop:
            opt = optax.rmsprop(lr_function)
        elif self.args.mode.optimizer.name == OptimizerKind.adam:
            opt = optax.adam(lr_function)
        elif self.args.mode.optimizer.name == OptimizerKind.adagrad:
            opt = optax.adagrad(lr_function)
        elif self.args.mode.optimizer.name == OptimizerKind.adadelta:
            opt = optax.adadelta(lr_function)
        elif self.args.mode.optimizer.name == OptimizerKind.lamb:
            opt = optax.lamb(lr_function)
        else:
            opt = optax.SGD(lr_function)


        return opt



    def _calculate_accuracy(self, network_dict, labels_dict, batch_reduce=True):
        ''' Calculate the accuracy.

            Images received here are not sparse but dense.
            This is to ensure equivalent metrics are computed for sparse and dense networks.

        '''


        # Predict the vertex, if needed:
        if self.args.network.vertex.active:
            network_dict['predicted_vertex'] = predict_vertex(network_dict, self.vertex_meta)

        return self.acc_calc(network_dict, labels_dict, batch_reduce)


    def summary(self, metrics, saver):

        if self._global_step % self.args.mode.summary_iteration == 0:
            for metric in metrics:
                name = metric
                value = metrics[metric]
                # if isinstance(value, torch.Tensor):
                #     # Cast metrics to 32 bit float
                #     value = value.float()
                saver.add_scalar(metric, value, self._global_step)




    def increment_global_step(self):

        self._global_step += 1



    def default_device_context(self):


        if self.args.run.compute_mode == ComputeMode.CUDA:
            return jax.default_device(self.default_device())
        elif self.args.run.compute_mode == ComputeMode.XPU:
            return contextlib.nullcontext()
        else:
            return contextlib.nullcontext()

    def default_device(self):

        if self.args.run.compute_mode == ComputeMode.CUDA:
            return jax.devices("gpu")[0]
        elif self.args.run.compute_mode == ComputeMode.XPU:
            device = torch.device("xpu")
        else:
            device = jax.devices('cpu')[0]
        return device

    def train_step(self, minibatch_data):

        # For a train step, we fetch data, run a forward and backward pass, and
        # if this is a logging step, we compute some logging metrics.

        global_start_time = datetime.datetime.now()

        # Reset the gradient values for this step:
        # self.opt.zero_grad()

        io_fetch_time = 0.0

        # Run the training step:
        with self.timing_context("train"):

            # Important: purge some parts of the input data to ensure compatible types:
            minibatch_data.pop("event_ids")
            state, metrics = self.function_lookup["train_step"](minibatch_data, self.train_state)
            self.train_state = state

        # Add the global step / second to the tensorboard log:
        try:
            metrics['global_step_per_sec'] = 1./self._seconds_per_global_step
            metrics['images_per_second'] = self.args.run.minibatch_size / self._seconds_per_global_step
        except:
            metrics['global_step_per_sec'] = 0.0
            metrics['images_per_second'] = 0.0

        metrics['io_fetch_time'] = io_fetch_time
        global_end_time = datetime.datetime.now()


        with self.timing_context("log"):
            self.log(metrics, self.log_keys, saver="train")


        with self.timing_context("summary"):

            # try to get the learning rate
            current_lr = self.lr_calculator(self.train_state.step)
            metrics["learning_rate"] = current_lr

            self.summary(metrics, saver=self.savers["train"])

            # Compute global step per second:
            self._seconds_per_global_step = (global_end_time - global_start_time).total_seconds()

            # Increment the global step value:
            self._global_step = self.train_state.step

        return

    def val_step(self, minibatch_data, store=True):

        # First, validation only occurs on training:
        if not self.is_training(): return

        if self.args.data.synthetic: return

        # perform a validation step
        # Validation steps can optionally accumulate over several minibatches, to
        # fit onto a gpu or other accelerator

        minibatch_data.pop("event_ids")
        metrics = self.function_lookup["val_step"](minibatch_data, self.train_state)


        if store:
            self.log(metrics, self.log_keys, saver="val")
            self.summary(metrics, saver=self.savers["val"])


        # Store these for hyperparameter logging.
        self.latest_metrics = metrics
        return metrics

    def finalize_val_metrics(self, metrics_list):
        metrics = {
            key : torch.mean(
                [ m[key] for m in metrics_list]
            ) for key in metrics_list[0].keys()
        }
        self.log(metrics, self.log_keys, saver="val")
        self.summary(metrics, saver=self.savers["val"])
        self.summary_images(network_dict["segmentation"], labels_dict["segmentation"], saver=self.savers["val"])


    def checkpoint(self):

    
        #  Don't checkpoint unless training:
        if not self.is_training(): return


        if self.args.run.run_units == RunUnit.iteration:

            if self._global_step % self.args.mode.checkpoint_iteration == 0 and self._global_step != 0:
                # Save a checkpoint, but don't do it on the first pass
                self.save_fn(self.train_state)
        else:
            # self._epoch % self.args.mode.checkpoint_iteration == 0 and
            if self._epoch_end:
                self.save_fn(self.train_state)

    def ana_step(self, batch):

        # First, validation only occurs on training:
        if self.is_training(): return

        # perform a validation step

        # Set network to eval mode
        self._net.eval()
        # self._net.train()


        # Run a forward pass of the model on the input image:
        with torch.no_grad():
            if self.args.run.precision == Precision.mixed and self.args.run.compute_mode == ComputeMode.CUDA:
                with torch.cuda.amp.autocast():
                    logits_dict, labels_dict = self.forward_pass(batch)
            else:
                logits_dict, labels_dict = self.forward_pass(batch)



        # If the input data has labels available, compute the metrics:
        if 'label' in batch:
            # Compute the loss
            # loss = self.loss_calculator(labels_dict, logits_dict)

            # Compute the metrics for this iteration:
            metrics = self._compute_metrics(logits_dict, labels_dict, loss_dict=None, batch_reduce=False)

            # We can count the number of neutrino id'd pixels per plane:
            n_neutrino_pixels = [ torch.sum(torch.argmax(p, axis=1) == 2, axis=(1,2)) for p in logits_dict["segmentation"]]
            predicted_vertex = predict_vertex(logits_dict, self.vertex_meta)
            predicted_label = torch.softmax(logits_dict["event_label"],axis=1)
            predicted_label = torch.argmax(predicted_label, axis=1)
            prediction_score = torch.max(predicted_label)
            # print(labels_dict['vertex'])
            additional_info = {
                "index"            : numpy.asarray(batch["entries"]),
                "event_id"         : numpy.asarray(batch["event_ids"]),
                "energy"           : batch["vertex"]["energy"],
                # "predicted_vertex" : predicted_vertex,
                "predicted_vertex0h" : predicted_vertex[:,0,0],
                "predicted_vertex0w" : predicted_vertex[:,0,1],
                "predicted_vertex1h" : predicted_vertex[:,1,0],
                "predicted_vertex1w" : predicted_vertex[:,1,1],
                "predicted_vertex2h" : predicted_vertex[:,2,0],
                "predicted_vertex2w" : predicted_vertex[:,2,1],
                # "predicted_vertex2" : predicted_vertex[:,2,:],
                # "true_vertex"      : labels_dict["vertex"]["xy_loc"],
                "true_vertex0h"      : labels_dict["vertex"]["xy_loc"][:,0,0],
                "true_vertex0w"      : labels_dict["vertex"]["xy_loc"][:,0,1],
                "true_vertex1h"      : labels_dict["vertex"]["xy_loc"][:,1,0],
                "true_vertex1w"      : labels_dict["vertex"]["xy_loc"][:,1,1],
                "true_vertex2h"      : labels_dict["vertex"]["xy_loc"][:,2,0],
                "true_vertex2w"      : labels_dict["vertex"]["xy_loc"][:,2,1],
                # "vertex_3dx"         : batch["vertex"]["xyz_loc"]["_x"],
                # "vertex_3dy"         : batch["vertex"]["xyz_loc"]["_y"],
                # "vertex_3dz"         : batch["vertex"]["xyz_loc"]["_z"],
                "N_neut_pixels0"     : n_neutrino_pixels[0],
                "N_neut_pixels1"     : n_neutrino_pixels[1],
                "N_neut_pixels2"     : n_neutrino_pixels[2],
                "predicted_label"  : predicted_label,
                "prediction_score"  : prediction_score,
                "true_label"       : labels_dict["event_label"],
            }

            # Move everything in the dictionary to CPU:
            additional_info.update(metrics)
            for key in additional_info.keys():
                if type(additional_info[key]) == torch.Tensor:
                    additional_info[key] = additional_info[key].cpu().numpy()

            self.local_df.append(pd.DataFrame.from_dict(additional_info))

            # Reduce the metrics over the batch size here:
            metrics = { key : torch.mean(metrics[key], axis=0) for key in metrics.keys() }

            self.accumulate_metrics(metrics)

            # print(minibatch_data)
            self.log(metrics, log_keys=self.log_keys, saver="ana")

        self._global_step += 1

        return

    def accumulate_metrics(self, metrics):

        self.inference_metrics['n'] += 1
        for key in metrics:
            if key not in self.inference_metrics:
                self.inference_metrics[key] = metrics[key]
                # self.inference_metrics[f"{key}_sq"] = metrics[key]**2
            else:
                self.inference_metrics[key] += metrics[key]
                # self.inference_metrics[f"{key}_sq"] += metrics[key]**2

    def inference_report(self):
        logger = logging.getLogger("CosmicTagger")

        if hasattr(self, "local_df") and self.local_df is not None:
            local_df = pd.concat(self.local_df)
            outdir = self.args.output_dir
            print(outdir)
            local_df.to_csv(f"{outdir}/rank_{self.rank}_{self.args.run.id}.csv")

        if not hasattr(self, "inference_metrics"):
            return
        n = self.inference_metrics["n"]
        total_entries = n*self.args.run.minibatch_size
        logger.info(f"Inference report: {n} batches processed for {total_entries} entries.")
        for key in self.inference_metrics:
            if key == 'n' or '_sq' in key: continue
            value = self.inference_metrics[key] / n
            logger.info(f"  {key}: {value:.4f}")

   