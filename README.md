[![Build Status](https://travis-ci.com/coreyjadams/CosmicTagger.svg?branch=master)](https://travis-ci.com/coreyjadams/CosmicTagger)



# Neutrino and Cosmic Tagging with UNet

This repository contains models and training utilities to train convolutional networks to separate cosmic pixels, background pixels, and neutrino pixels in a neutrinos dataset.  There are several variations.

This network is implemented in both torch and tensorflow.  To select between the networks, you can use the `--framework` parameter.  It accepts either `tensorflow` or `torch`.  The model is available in a development version with sparse convolutions in the `torch` framework.

## Configuration

In general, this network has a suite of parameters available.  For example, running the torch version will print the following configuration:

```
-- CONFIG --
 AUX_FILE......................: None
 AUX_IO_VERBOSITY..............: 3
 AUX_ITERATION.................: 10
 AUX_MINIBATCH_SIZE............: 2
 BALANCE_LOSS..................: True
 BATCH_NORM....................: True
 BLOCKS_DEEPEST_LAYER..........: 2
 BLOCKS_FINAL..................: 2
 BLOCKS_PER_LAYER..............: 2
 BLOCK_CONCAT..................: False
 CHECKPOINT_DIRECTORY..........: None
 CHECKPOINT_ITERATION..........: 100
 COMPUTE_MODE..................: CPU
 CONNECTIONS...................: concat
 CONV_MODE.....................: 2D
 DATA_FORMAT...................: channels_first
 DISTRIBUTED...................: False
 DOWNSAMPLE_IMAGES.............: 1
 DOWNSAMPLING..................: convolutional
 FILE..........................: /Users/corey.adams/data/dlp_larcv3/sbnd_cosmic_samples/cosmic_tagging_downsample_train_sparse.h5
 FRAMEWORK.....................: torch
 GRADIENT_ACCUMULATION.........: 1
 GROWTH_RATE...................: multiplicative
 IMAGE_PRODUCER................: sbndwire
 INPUT_HALF_PRECISION..........: False
 INTER_OP_PARALLELISM_THREADS..: 4
 INTRA_OP_PARALLELISM_THREADS..: 64
 IO_VERBOSITY..................: 3
 ITERATIONS....................: 4
 LABEL_PRODUCER................: sbnd_cosmicseg
 LEARNING_RATE.................: 0.0003
 LOGGING_ITERATION.............: 1
 LOG_DIRECTORY.................: ./log
 LOSS_SCALE....................: 1.0
 MINIBATCH_SIZE................: 1
 MODE..........................: train
 MODEL_HALF_PRECISION..........: False
 NETWORK_DEPTH.................: 2
 N_INITIAL_FILTERS.............: 1
 OPTIMIZER.....................: Adam
 REGULARIZE_WEIGHTS............: 0.0001
 RESIDUAL......................: False
 SPARSE........................: False
 SUMMARY_ITERATION.............: 1
 SYNTHETIC.....................: False
 TRAINING......................: True
 UPSAMPLING....................: convolutional
 USE_BIAS......................: True
 VERBOSITY.....................: 0

```

The parameters are all controlled via argparse, and so are easy to inspect.  They can be viewed in src/utils/flags.py, or via command line via the -help flag.

## Datasets

The data for this network is in larcv3 format (https://github.com/DeepLearnPhysics/larcv3).  Currently, data is available in full resolution (HxW == 1280x2048) of 3 images per training sample.  This image size is large, and the network is large, so to accomodate older hardware or smaller GPUs this can be run with a reduced image size.  The datasets are kept at full resolution but a downsampling operation is applied prior to feeding images and labels to the network.

The UNet design is symmetric and does downsampling/upsampling by factors of 2.  So, in order to preserve the proper sizes during the upsampling sets, it's important that the smallest resolution image reached by the network is not odd dimensions.  Concretely, this means that the sum of `NETWORK_DEPTH` and `DOWNSAMPLE_IMAGES` must be less than 8.

The training dataset is 43075 images.  The testing set, used to monitor overfitting during training, is 7362 images.  The validation set is O(15k) images.

### Data format

The dataformat for these images is sparse.  Images are stored as a flat array of indexes and values, where index is mapped to a 2D index that is unraveled to a coordinate pair.  Each image is stored consecutively in file, and because of the non-uniform size of the sparse data there are mapping algorithms to go into the file, read the correct sequence of (index, value) pairs, and convert to (image, x, y, value) tuples.

During training, memory will buffer for each minibatch in a current and next buffer.  Since each image is not uniform size, the memory buffer is slightly larger than the largest image in the datasets.  For the fullres data, this is about 50k pixels.  Larcv3 handles reading from file and buffering into memory.

In distributed mode, each worker will read it's own data from the central file, and the entries to read in are coordinated by the master rank.

## Frameworks

### Tensorflow

With tensorflow, the model is available and implemented with 2D convolutions and 3D convolutions.  The 3D convolution implementation differs slightly from the 2D implementation: at the deepest layer, the 2D implementation concatenates across planes, and then performs shared convolutions.  The 3D implementation uses convolutions of [1,3,3] to emulate 2D convolutions throughout the network, but at the deepest layer uses [3,3,3] convolutions instead.
 
### Pytorch

This model is available in pytorch on the branch `torch`.  As much as possible, the structure of the model is identical to the tensorflow model.  Like the tensorflow models, the 3D model in pytorch is slightly different from the 2D model.

### Sparse Pytorch

The sparse implementation of this network requires sparsehash, and SparseConvNet.  The sparse pytorch model is equivalent to the 3D pytorch model, and the core of the network is done with sparse convolutions.  The final step, the bottleneck operation, is done by converting the sparse activations to dense, and applying a single bottleneck layer to the dense activations.  This allows the network to quickly and accurately predict background pixels, without carrying they through the network.

# Running the software

In all cases, there is a general python executable in `bin/exec.py`.  This takes several important arguments and many minor arguments.  Important arguments are:

`python bin/exec.py mode [-d] --file /path/to/file.h5 -i 100 -mb 8 `

mode is either `train` or `iotest` (or inference, but untested in this repo).  `-d` toggles distributed training, which will work even on one node and if python is executed by mpirun or aprun, will work.  `-i` is the number of iterations, and `-mb` is the minibatch size.  All other arguments can be seen in by calling `python bin/exec.py --help`

This is a memory intesive network with the dense models.  Typically, 1 image in the standard network can utilize more than 10GB of memory to store intermediate activations.  To allow increased batch size, both `torch` and `tf` models support gradient accumulation across several images before weight updates.  Set the `--gradient-accumulation` flag to an integer greater than 1 to enable this.

# Analysis Metrics

There are several analysis metrics that are used to judge the quality of the training:
 1) Overall Accuracy of Segmentation labels. Each pixel should be labeled as cosmic, neutrino, or background.  Because the images are very sparse, this metric should easily exceed 99.9%+ accuracy.
 2) Non-background Accuracy: of all pixels with a label != bkg, what is the accuracy? This should acheive > 95%
 3) Cosmic IoU: what is the IoU of all pixels predicted cosmic and all pixels labeled cosmic?  This should acheive > 90%
 4) Neutrino IoU: Same definition as 4 but for neutrinos.  This should acheive > 70%.
