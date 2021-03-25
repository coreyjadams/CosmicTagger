import sys, os
import random

from omegaconf import OmegaConf


for i in range(10):

    # Load the default config:
    conf = OmegaConf.load('src/config/config.yaml')

    output_config_name = f"src/config/image_scaling_configs/config{i}.yaml"


    # Generate a suite of config options:

    configuration = {
      "defaults" :
        [ {"network": "uresnet"},
          {"framework": "tensorflow"},
          {"mode": "training"},
          {"data": "real"},
        ],
      "run": {
          "distributed"        : False,
          "compute_mode"       : "GPU",
          "iterations"         : 50,
          "minibatch_size"     : 2,
          "aux_minibatch_size" : "${run.minibatch_size}",
          "aux_iterations"     : 10,
          "id"                 : "???",
          "precision"          : "float32",
          "profile"            : False,
          "output_dir"         : "output/${framework.name}/${network.name}/${run.id}/",
        }
    }

    # Minibatch size must be an even multiple of 16, and at least 64 but less than 128

    mb = random.choice([128,192])

    conf['compute_mode'] = {'minibatch_size' : mb}

    # Make sure we run for 25 epochs:
    conf['run']['iterations'] = int(25 * 43075 / mb)

    # Network parameters:
    conf['network'] = {}
    conf['network']['bias'] = random.choice([True, False])
    conf['network']['batch_norm'] = random.choice([True, False])
    conf['network']['n_initial_filters'] = 8*random.randint(1,4)
    conf['network']['blocks_per_layer'] = random.randint(1,3)
    conf['network']['blocks_deepest_layer'] = random.randint(1,5)
    conf['network']['blocks_final'] = random.randint(0,5)
    conf['network']['bottleneck_deepest'] = 8*random.randint(1,32)
    conf['network']['residual'] = random.choice([True, False])
    conf['network']['downsampling'] = random.choice(["max_pooling", "convolutional"]) 
    conf['network']['upsampling'] =  random.choice(["interpolation", "convolutional"]) 
    conf['network']['connections'] = random.choice(["concat", "sum"])

    conf['mode'] = {}
    conf['mode']['optimizer'] = {}
    conf['mode']['optimizer']['learning_rate'] = 10.**random.uniform(-3.5, -2.5)
    conf['mode']['optimizer']['loss_balance_scheme'] = random.choice(["none", "light", "focal"])



    OmegaConf.save(conf, output_config_name)
