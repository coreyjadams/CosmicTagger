[![Build Status](https://travis-ci.com/coreyjadams/CosmicTagger.svg?branch=master)](https://travis-ci.com/coreyjadams/CosmicTagger)



# Neutrino and Cosmic Tagging with UNet

This repository contains models and training utilities to train convolutional networks to separate cosmic pixels, background pixels, and neutrino pixels in a neutrinos dataset.  There are several variations. A detailed description of the code can be found in:
* [*Cosmic Background Removal with Deep Neural Networks in SBND*](https://arxiv.org/abs/2012.01301) 


This network is implemented in both PyTorch and TensorFlow.  To select between the networks, you can use the `--framework` parameter.  It accepts either `tensorflow` or `torch`.  The model is available in a development version with sparse convolutions in the `torch` framework.

## Installation

CosmicTagger's dependencies can be installed via Conda and/or Pip. For example, Conda can be used to acquire many of the build dependencies for both CosmicTagger and `larcv3`

```
conda create -n cosmic_tagger python==3.7
conda install cmake hdf5 scikit-build numpy
```

As of April 2021, the version of `larcv3` on PyPI (v3.3.3) does not work with CosmicTagger. A version corresponding to commit `c73936e` or later is currently necessary. To build `larcv3` from source, 
```
git clone https://github.com/DeepLearnPhysics/larcv3.git
cd larcv3
git submodule update --init
pip install -e .
```

Then, in the CosmicTagger directory, 
```
pip install -r requirements.txt
```

## Configuration

In general, this network has a suite of parameters available, for example:

```
-- CONFIG --
data:
  aux_file....................: cosmic_tagging_test.h5
  data_directory..............: /grand/projects/datascience/cadams/datasets/SBND/
  data_format.................: channels_last
  downsample..................: 1
  file........................: cosmic_tagging_train.h5
  synthetic...................: False
framework:
  environment_variables:
    TF_XLA_FLAGS..............: --tf_xla_auto_jit=2
  inter_op_parallelism_threads: 2
  intra_op_parallelism_threads: 24
  name........................: tensorflow
mode:
  checkpoint_iteration........: 500
  logging_iteration...........: 1
  name........................: train
  no_summary_images...........: False
  optimizer:
    gradient_accumulation.....: 1
    learning_rate.............: 0.0003
    loss_balance_scheme.......: light
    name......................: adam
  summary_iteration...........: 1
  weights_location............:
network:
  batch_norm..................: True
  bias........................: True
  block_concat................: False
  blocks_deepest_layer........: 5
  blocks_final................: 5
  blocks_per_layer............: 2
  bottleneck_deepest..........: 256
  connections.................: concat
  conv_mode...................: 2D
  data_format.................: channels_last
  downsampling................: max_pooling
  filter_size_deepest.........: 5
  growth_rate.................: 1
  n_initial_filters...........: 16
  name........................: uresnet
  network_depth...............: 6
  residual....................: True
  upsampling..................: interpolation
  weight_decay................: 0.0
run:
  aux_iterations..............: 10
  aux_minibatch_size..........: 16
  compute_mode................: GPU
  distributed.................: False
  id..........................: test
  iterations..................: 50
  minibatch_size..............: 16
  output_dir..................: output/tensorflow/uresnet/test/
  precision...................: float32
  profile.....................: False
```

## Datasets

Data may be downloaded from Globus  [here](https://app.globus.org/file-manager?origin_id=d02b81ca-6d77-4e41-a4c5-6161cf5c3bcb&origin_path=%2F).

The data for this network is in larcv3 format (https://github.com/DeepLearnPhysics/larcv3).  Currently, data is available in full resolution (HxW == 1280x2048) of 3 images per training sample.  This image size is large, and the network is large, so to accomodate older hardware or smaller GPUs this can be run with a reduced image size.  The datasets are kept at full resolution but a downsampling operation is applied prior to feeding images and labels to the network.

The UNet design is symmetric and does downsampling/upsampling by factors of 2.  So, in order to preserve the proper sizes during the upsampling sets, it's important that the smallest resolution image reached by the network does not contain a dimension with an odd number of pixels.  Concretely, this means that the sum of `network_depth` and `downsample_images` must be less than 8, since 1280 pixels / 2^8 = 5. 

The training dataset `cosmic_tagging_train.h5` contains 43075 images.  The validation set `cosmic_tagging_val.h5`, specified by `--aux-file` and used to monitor overfitting during training, is 7362 images.  The final hold-out test set `cosmic_tagging_test.h5` contains 7449 images. To evaluate the accuracy of a trained model on the hold-out test set (after all training and tuning is complete), rerun the application in inference mode with `data.file=cosmic_tagging_test.h5`

### Data format

The dataformat for these images is sparse.  Images are stored as a flat array of indexes and values, where index is mapped to a 2D index that is unraveled to a coordinate pair.  Each image is stored consecutively in file, and because of the non-uniform size of the sparse data there are mapping algorithms to go into the file, read the correct sequence of (index, value) pairs, and convert to (image, x, y, value) tuples.

During training, memory will buffer for each minibatch in a current and next buffer.  Since each image is not uniform size, the memory buffer is slightly larger than the largest image in the datasets.  For the fullres data, this is about 50k pixels.  Larcv3 handles reading from file and buffering into memory.

In distributed mode, each worker will read it's own data from the central file, and the entries to read in are coordinated by the master rank.

## Frameworks

### TensorFlow

With TensorFlow, the model is available and implemented with 2D convolutions and 3D convolutions.  The 3D convolution implementation differs slightly from the 2D implementation: at the deepest layer, the 2D implementation concatenates across planes, and then performs shared convolutions.  The 3D implementation uses convolutions of [1,3,3] to emulate 2D convolutions throughout the network, but at the deepest layer uses [3,3,3] convolutions instead.

### PyTorch

As much as possible, the structure of the model is identical to the TensorFlow model.  Like the TensorFlow models, the 3D model in PyTorch is slightly different from the 2D model.

### Sparse PyTorch

The sparse implementation of this network requires sparsehash, and SparseConvNet.  The sparse PyTorch model is equivalent to the 3D PyTorch model, and the core of the network is done with sparse convolutions.  The final step, the bottleneck operation, is done by converting the sparse activations to dense, and applying a single bottleneck layer to the dense activations.  This allows the network to quickly and accurately predict background pixels, without carrying they through the network.

# Running the software

In all cases, there is a general Python executable in `bin/exec.py`.  This takes several important arguments and many minor arguments.  Important arguments are:

`python bin/exec.py mode=[iotest|train|inference] run.id=[run-id] [other arguments]`

mode is either `train` or `iotest` or `inference`.  `run.distributed=true` toggles distributed training, which will work even on one node and if python is executed by mpirun (or similar), will work.  `run.iterations=$ITER` is the number of iterations, and `run.minibatch_size` is the minibatch size.  All other arguments can be seen in by calling `python bin/exec.py --help`.  In general, you can override an argument by setting it on the command line, and nested arguments are seperated with a period.  For example: `mode.optimizer.learning_rate=123.456` is valid (but won't converged of course) while just `learning_rate=123.456` will be an error.

This is a memory intesive network with the dense models.  Typically, 1 image in the standard network can utilize more than 10GB of memory to store intermediate activations.  To allow increased batch size, both `torch` and `tensorflow` models support gradient accumulation across several images before weight updates.  Set the `mode.optimizer.gradient_accumulation` flag to an integer greater than 1 to enable this.

# Analysis Metrics

There are several analysis metrics that are used to judge the quality of the training:
 1) Overall Accuracy of Segmentation labels. Each pixel should be labeled as cosmic, neutrino, or background.  Because the images are very sparse, this metric should easily exceed 99.9%+ accuracy.
 2) Non-background Accuracy: of all pixels with a label != bkg, what is the accuracy? This should acheive > 95%
 3) Cosmic IoU: what is the IoU of all pixels predicted cosmic and all pixels labeled cosmic?  This should acheive > 90%
 4) Neutrino IoU: Same definition as 4 but for neutrinos.  This should acheive > 70%.


# Citations

```
@misc{sbndcollaboration2021cosmic,
      title={Cosmic Background Removal with Deep Neural Networks in SBND}, 
      author={SBND Collaboration and R. Acciarri and C. Adams and C. Andreopoulos and J. Asaadi and M. Babicz and C. Backhouse and W. Badgett and L. Bagby and D. Barker and V. Basque and M. and C. and Q. Bazetto and M. Betancourt and A. Bhanderi and A. Bhat and C. Bonifazi and D. Brailsford and A. and G. Brandt and T. Brooks and M. and F. Carneiro and Y. Chen and H. Chen and G. Chisnall and J. and I. Crespo-Anadón and E. Cristaldo and C. Cuesta and I. and L. de Icaza Astiz and A. De Roeck and G. de Sá Pereira and M. Del Tutto and V. Di Benedetto and A. Ereditato and J. and J. Evans and A. and C. Ezeribe and R. and S. Fitzpatrick and B. and T. Fleming and W. Foreman and D. Franco and I. Furic and A. and P. Furmanski and S. Gao and D. Garcia-Gamez and H. Frandini and G. Ge and I. Gil-Botella and S. Gollapinni and O. Goodwin and P. Green and W. and C. Griffith and R. Guenette and P. Guzowski and T. Ham and J. Henzerling and A. Holin and B. Howard and R. and S. Jones and D. Kalra and G. Karagiorgi and L. Kashur and W. Ketchum and M. and J. Kim and V. and A. Kudryavtsev and J. Larkin and H. Lay and I. Lepetic and B. and R. Littlejohn and W. and C. Louis and A. and A. Machado and M. Malek and D. Mardsen and C. Mariani and F. Marinho and A. Mastbaum and K. Mavrokoridis and N. McConkey and V. Meddage and D. and P. Méndez and T. Mettler and K. Mistry and A. Mogan and J. Molina and M. Mooney and L. Mora and C. and A. Moura and J. Mousseau and A. Navrer-Agasson and F. and J. Nicolas-Arnaldos and J. and A. Nowak and O. Palamara and V. Pandey and J. Pater and L. Paulucci and V. and L. Pimentel and F. Psihas and G. Putnam and X. Qian and E. Raguzin and H. Ray and M. Reggiani-Guzzo and D. Rivera and M. Roda and M. Ross-Lonergan and G. Scanavini and A. Scarff and D. and W. Schmitz and A. Schukraft and E. Segreto and M. Soares Nunes and M. Soderberg and S. Söldner-Rembold and J. Spitz and N. and J. and C. Spooner and M. Stancari and G. and V. Stenico and A. Szelc and W. Tang and J. Tena Vidal and D. Torretta and M. Toups and C. Touramanis and M. Tripathi and S. Tufanli and E. Tyley and G. and A. Valdiviesso and E. Worcester and M. Worcester and G. Yarbrough and J. Yu and B. Zamorano and J. Zennamo and A. Zglam},
      year={2021},
      eprint={2012.01301},
      archivePrefix={arXiv},
      primaryClass={physics.data-an}
}
```

