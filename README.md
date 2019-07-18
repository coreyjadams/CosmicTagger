# Neutrino and Cosmic Tagging with UNet

This repository contains models and training utilities to train convolutional networks to separate cosmic pixels, background pixels, and neutrino pixels in a neutrinos dataset.  There are several variations.

In general, this network has a suite of parameters available.  For example, running the torch version on theta for debugging, the parameters I used were:

```
-- CONFIG --
 MINIBATCH_SIZE = 1
 LOG_DIRECTORY = '/home/cadams/Theta/DLP3/CosmicTagger-torch//log/torch/nodes_test_conda/'
 FILE = '/projects/datascience/cadams/datasets/SBND/H5/cosmic_tagging_downsample/cosmic_tagging_downsample_train_sparse.h5'
 LABEL_PRODUCER = 'sbnd_cosmicseg'
 ITERATIONS = 10000
 USE_BIAS = True
 N_INITIAL_FILTERS = 12
 CHECKPOINT_ITERATION = 500
 RESIDUAL = True
 LEARNING_RATE = 0.0001
 LOSS_SCALE = 1.0
 IO_VERBOSITY = 3
 MAX_VOXELS = 35000
 GROWTH_RATE = 'multiplicative'
 BLOCK_CONCAT = False
 MODE = 'train'
 BALANCE_LOSS = False
 NETWORK_DEPTH = 5
 TRAINING = True
 MODEL_HALF_PRECISION = False
 AUX_ITERATION = 10
 REGULARIZE_WEIGHTS = 0.0001
 RES_BLOCKS_DEEPEST_LAYER = 4
 AUX_IO_VERBOSITY = 3
 RES_BLOCKS_FINAL = 2
 SPARSE = False
 CONNECTIONS = 'sum'
 INTRA_OP_PARALLELISM_THREADS = 128
 COMPUTE_MODE = 'CPU'
 SHARE_WEIGHTS = True
 BOTTLENECK_SIZE = -1
 INPUT_HALF_PRECISION = False
 RES_BLOCKS_PER_LAYER = 2
 DISTRIBUTED = False
 OPTIMIZER = 'adam'
 CHECKPOINT_DIRECTORY = None
 SUMMARY_ITERATION = 1
 CONNECT_PRE_RES_BLOCKS_UP = True
 SHAPE = [640, 1024]
 CONV_MODE = '2D'
 VERBOSITY = 0
 AUX_MINIBATCH_SIZE = 1
 LOGGING_ITERATION = 1
 AUX_FILE = '/projects/datascience/cadams/datasets/SBND/H5/cosmic_tagging_downsample/cosmic_tagging_downsample_test_sparse.h5'
 BATCH_NORM = True
 CONNECT_PRE_RES_BLOCKS_DOWN = True
 IMAGE_PRODUCER = 'sbndwire'
 NPLANES = 3
 INTER_OP_PARALLELISM_THREADS = 2
```

The parameters are all controlled via argparse, and so are easy to inspect.  They can be viewed in src/utils/flags.py.

## Datasets

The data for this network is in larcv3 format (https://github.com/DeepLearnPhysics/larcv3).  Currently, data is available in a downsampled version (HxW == 640x1024) of 3 images per training sample.  Full resolution data is being processed and will be available very soon.  On Theta, data is available at /lus/theta-fs0/projects/datascience/dl_workloads/neutrinos/datasets/downsample/ and ../fullres/.

The training dataset is 43075 images.  The testing set, used to monitor overfitting during training, is 7362 images.  The validation set is O(15k) images.

### Data format

The dataformat for these images is sparse.  Images are stored as a flat array of indexes and values, where index is mapped to a 2D index that is unraveled to a coordinate pair.  Each image is stored consecutively in file, and because of the non-uniform size of the sparse data there are mapping algorithms to go into the file, read the correct sequence of (index, value) pairs, and convert to (image, x, y, value) tuples.

During training, memory buffers for each minibatch in several memory buffers, filled with several memory threads.  This attempts to hide IO latency from disk.  Since each image is not uniform size, the memory buffer is slightly larger than the largest image in the datasets.  For the fullres data, this is about 80k.  For downsampled data, this is about 35k.

Each minibatch is loaded by a root rank for distributed training, and the uniform size buffers are scattered to each node.  Each rank unpacks its data into a full resolution image or an appropriate sparse image for the sparse training.


## Tensorflow

This model is available in tensorflow on the branch `tf`.  It has been tested with tf 1.13.1 on Theta, Cooley, and V100 nodes on JLSE.

On Theta, tensorflow was installed via conda (default channel) and by using the datascience packages.  On Cooley, it was installed via conda as well as via pip inside of a singularity container.

With the default parameters and 12 initial filters, the model has approximately 100M parameters.  By default, batch norm is used in this model but it is not used on Theta for severe performance issues.  This is under investigation.

Distributed training is enabled with the `-d` flag and works on Theta, Cooley, and DGX@JLSE.

On Cooley, conda and singularity have indistinguishable performance for both single node and distributed learning.  Benchmarks for performance were performed up to 14nodes, or 28 GPUs, with a throughput ratio 

On Theta, single node performance was checked with both the datascience and conda installations.  For the same model as on Cooley, total image throughput (img/s) with batch_size of 1 is:
 - 0.046 Img/s (datascience tensorflow)
 - 0.086 Img/s (conda tensorflow)

For distributed training, the image throughput is:
 - 
 - 0.59 Img/s (conda tensorflow, 85% scaling efficiency from single node)

On Cooley, using 1 image per GPU, 2 GPUs per node, throughput scales as follows:
 - 0.24 Img/s (1 node)
 - 0.49 Img/s (2 nodes)
 - 0.95 Img/s (4 nodes)
 - 1.82 Img/s (8 nodes)
 - 2.92 Img/s (14 nodes)
 - 3.7  Img/s (14 nodes @ 2 events/GPU)

It should be noted that higher total throughput is likely possible on Theta using more images per node (currently 1).  Other parameters are set as "default" for performance:
```
export KMP_AFFINITY=granularity=fine,verbose,compact 
export OMP_NUM_THREADS=64
export KMP_BLOCKTIME=0
export MKLDNN_VERBOSE=0 
export MPICH_MAX_THREAD_SAFETY=multiple

aprun -n ${N_NODES} -N 1 \
-cc depth \
-j 1 \
-d 64 \
python ...
```
Interop is set to 4, intra op is set to 64.

## Pytorch

This model is available in pytorch on the branch `torch`.  As much as possible, the structure of the model is identical to the tensorflow model.  The use of bias in convolution operations introduces a small discrepancy in number of parameters.

The pytorch implementation also works on Theta, Cooley, JLSE.  There is a sparse and dense implementation for pytorch.

**NOTE: pytorch has only a dense implementation available on Theta at this second, but the software installation is easy and will be done soon**

On Theta, single node performance (Using batch norm - indicating my tf implementation is using a suboptimal BN op) is as follows:
 - 0.062 (datascience pytorch 1.1)
 - 0.051 (conda pytorch 1.1)

Pytorch jobs with distributed training are under study.  It has run successfully on Cooley and JLSE.

### Sparse Pytorch

The sparse implementation of this network requires sparsehash, and SparseConvNet.  It is installed and available on Cooley and JLSE, theta installation forthcoming very soon.

# Running the software

Example scripts are provided for each of the different installations available.

In all cases, there is a general python executable in `bin/exec.py`.  This takes several important arguments and many minor arguments.  Important arguments are:

`python bin/exec.py mode [-d] [--sparse] --file /path/to/file.h5 -i 100 -mb 8 `

mode is either `train` or `iotest` (or inference, but untested in this repo).  `-d` toggles distributed training, which will work even on one node and if python is executed by mpirun or aprun, will work.  `-i` is the number of iterations, and `-mb` is the minibatch size.  All other arguments can be seen in by calling `python bin/exec.py --help`

# Analysis Metrics

There are several analysis metrics that are used to judge the quality of the training:
 1) Overall Accuracy of Segmentation labels. Each pixel should be labeled as cosmic, neutrino, or background.  Because the images are very sparse, this metric should easily exceed 99.9% accuracy.
 2) Non-background Accuracy: of all pixels with a label != bkg, what is the accuracy? This should acheive > 90%
 3) Neutrino Accuracy: of all pixels with label == neutrino, what is the accuracy?  This should acheive > 90%, though is an ill-posed question for some interactions where the neutrino did not deposit energy.
 4) Cosmic IoU: what is the IoU of all pixels predicted cosmic and all pixels labeled cosmic?  This should acheive > 70%
 5) Neutrino IoU: Same definition as 4 but for neutrinos.  This should acheive > 70%.
