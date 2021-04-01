[![Build Status](https://travis-ci.com/coreyjadams/CosmicTagger.svg?branch=master)](https://travis-ci.com/coreyjadams/CosmicTagger)



# Neutrino and Cosmic Tagging with UNet

This repository contains models and training utilities to train convolutional networks to separate cosmic pixels, background pixels, and neutrino pixels in a neutrinos dataset.  There are several variations.

This network is implemented in both torch and tensorflow.  To select between the networks, you can use the `--framework` parameter.  It accepts either `tensorflow` or `torch`.  The model is available in a development version with sparse convolutions in the `torch` framework.

## Installation

CosmicTagger's dependencies can be installed via Conda and/or Pip. For example, Conda can be used to acquire many of the build dependencies for both CosmicTagger and `larcv3`

```
conda create -n cosmic_tagger python==3.7
conda install cmake h5py scikit-build numpy
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

In general, this network has a suite of parameters available.  For example, running the `--framework torch` version will print the following configuration:

```
-- CONFIG --
 aux_file......................: /lus/theta-fs0/projects/datascience/cadams/datasets/SBND/H5/cosmic_tagging/cosmic_tagging_test.h5
 aux_iteration.................: 10
 aux_minibatch_size............: 2
 batch_norm....................: True
 block_concat..................: False
 blocks_deepest_layer..........: 5
 blocks_final..................: 0
 blocks_per_layer..............: 2
 bottleneck_deepest............: 256
 checkpoint_directory..........: None
 checkpoint_iteration..........: 500
 compute_mode..................: GPU
 connections...................: sum
 conv_mode.....................: 2D
 data_format...................: channels_first
 distributed...................: False
 distributed_mode..............: horovod
 downsample_images.............: 1
 downsampling..................: max_pooling
 file..........................: /lus/theta-fs0/projects/datascience/cadams/datasets/SBND/H5/cosmic_tagging/cosmic_tagging_train.h5
 filter_size_deepest...........: 5
 framework.....................: torch
 gradient_accumulation.........: 1
 growth_rate...................: additive
 inter_op_parallelism_threads..: 4
 intra_op_parallelism_threads..: 24
 iterations....................: 25000
 learning_rate.................: 0.0003
 log_directory.................: log/
 logging_iteration.............: 1
 loss_balance_scheme...........: focal
 minibatch_size................: 2
 mode..........................: train
 n_initial_filters.............: 16
 network_depth.................: 6
 no_summary_images.............: False
 optimizer.....................: rmsprop
 precision.....................: float32
 profile.......................: False
 regularize_weights............: 1e-05
 residual......................: True
 sparse........................: False
 start_index...................: 0
 summary_iteration.............: 1
 synthetic.....................: False
 training......................: True
 upsampling....................: interpolation
 use_bias......................: True
 weight_decay..................: 0.0

```

The parameters are all controlled via argparse, and so are easy to inspect.  They can be viewed in `src/utils/flags.py`, or on the command line via the `-help` flag.

## Datasets

The data for this network is in larcv3 format (https://github.com/DeepLearnPhysics/larcv3).  Currently, data is available in full resolution (HxW == 1280x2048) of 3 images per training sample.  This image size is large, and the network is large, so to accomodate older hardware or smaller GPUs this can be run with a reduced image size.  The datasets are kept at full resolution but a downsampling operation is applied prior to feeding images and labels to the network.

The UNet design is symmetric and does downsampling/upsampling by factors of 2.  So, in order to preserve the proper sizes during the upsampling sets, it's important that the smallest resolution image reached by the network does not contain a dimension with an odd number of pixels.  Concretely, this means that the sum of `network_depth` and `downsample_images` must be less than 8, since 1280 pixels / 2^8 = 5. 

The training dataset `cosmic_tagging_train.h5` is 43075 images.  The testing set `cosmic_tagging_test.h5`, specified by `--aux-file` and used to monitor overfitting during training, is 7362 images.  The validation set `cosmic_tagging_val.h5` contains 7449 images.

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

`python bin/exec.py mode [-d] --file /path/to/file.h5 -i 100 -mb 8 `

mode is either `train` or `iotest` (or inference, but untested in this repo).  `-d` toggles distributed training, which will work even on one node and if python is executed by mpirun or aprun, will work.  `-i` is the number of iterations, and `-mb` is the minibatch size.  All other arguments can be seen in by calling `python bin/exec.py --help`

This is a memory intesive network with the dense models.  Typically, 1 image in the standard network can utilize more than 10GB of memory to store intermediate activations.  To allow increased batch size, both `torch` and `tf` models support gradient accumulation across several images before weight updates.  Set the `--gradient-accumulation` flag to an integer greater than 1 to enable this.

# Analysis Metrics

There are several analysis metrics that are used to judge the quality of the training:
 1) Overall Accuracy of Segmentation labels. Each pixel should be labeled as cosmic, neutrino, or background.  Because the images are very sparse, this metric should easily exceed 99.9%+ accuracy.
 2) Non-background Accuracy: of all pixels with a label != bkg, what is the accuracy? This should acheive > 95%
 3) Cosmic IoU: what is the IoU of all pixels predicted cosmic and all pixels labeled cosmic?  This should acheive > 90%
 4) Neutrino IoU: Same definition as 4 but for neutrinos.  This should acheive > 70%.
