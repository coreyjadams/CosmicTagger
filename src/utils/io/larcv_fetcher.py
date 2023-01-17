import os
import time

from . import data_transforms
import tempfile

import numpy

import string, random

import logging
logger = logging.getLogger("cosmictagger")

from src.config import DataFormatKind


# Functional programming approach to building up the dataset objects:

def meta(dataset_version):
    if dataset_version == 1:
        return numpy.array([
            ([1280, 2048], [1280, 2048],[0.,0.]),
            ([1280, 2048], [1280, 2048],[0.,0.]),
            ([1280, 2048], [1280, 2048],[0.,0.])],
            dtype=[
                ('full_pixels', "int", (2)),
                ('size', "float", (2)),
                ('origin', "float", (2)),
            ]
        )
    elif dataset_version == 2:
        return numpy.array([
            ([1408, 2048], [439.4660061919505, 614.4],[-18.1030030959, -9.6]),
            ([1408, 2048], [439.4660061919505, 614.4],[-18.1030030959, -9.6]),
            ([1408, 2048], [439.4660061919505, 614.4],[-18.1030030959, -57.5999999])],
            dtype=[
                ('full_pixels', "int", (2)),
                ('size', "float", (2)),
                ('origin', "float", (2)),
            ]
        )

def image_shape(meta, downsample):
    """ downsample is the full number, aka 2**2 = 4"""

    return [ int(i / downsample) for i in meta['full_pixels'][0] ]


def batch_dims(data_format, meta, downsample, batch_size):

    i_shape = image_shape(meta, downsample)

    if data_format == DataFormatKind.channels_first:
        shape = [batch_size, 3, i_shape[0], i_shape[1]]
    elif data_format == DataFormatKind.channels_last:
        shape = [batch_size, i_shape[0], i_shape[1], 3]
    else:
        raise Exception(f"Unknown data format: {args.data.data_format}")
    return shape

def create_larcv_interface(random_access_mode, distributed, seed):

    # Not needed, enforced by data.py
    # if random_access_mode not in ["serial_access", "random_blocks"]: 
    #     raise Exception(f"Can not use mode {random_access_mode}")

    if seed == -1:
        seed = int(time.time())
    if distributed:
        from larcv import distributed_queue_interface as queueloader
    else:
        from larcv import queueloader


    larcv_interface = queueloader.queue_interface(
        random_access_mode=str(random_access_mode.name), seed=seed)
    larcv_interface.no_warnings()

    return larcv_interface

def prepare_cosmic_tagger_config(batch_size, input_file, data_args, name,
                                 event_id=False, vertex_depth=None):
    

    # First, verify the files exist:
    if not os.path.exists(input_file):
        raise Exception(f"File {input_file} not found")


    downsample = 2**data_args.downsample

    from larcv.config_builder import ConfigBuilder
    cb = ConfigBuilder()
    cb.set_parameter([str(input_file)], "InputFiles")
    cb.set_parameter(5, "ProcessDriver", "IOManager", "Verbosity")
    cb.set_parameter(5, "ProcessDriver", "Verbosity")
    cb.set_parameter(5, "Verbosity")

    # Do we need to do compression?
    if data_args.downsample != 0:
        cb.add_preprocess(
            datatype = "sparse2d",
            Product  = "sparse2d",
            producer = "sbndwire",
            process  = "Downsample",
            OutputProducer = "sbndwire",
            Downsample = downsample,
            PoolType = 1 # average,
        )
        cb.add_preprocess(
            datatype = "sparse2d",
            Product  = "sparse2d",
            producer = "sbnd_cosmicseg",
            process  = "Downsample",
            OutputProducer = "sbnd_cosmicseg",
            Downsample = downsample,
            PoolType = 2 # max
        )
    # Bring in the wires:
    cb.add_batch_filler(
        datatype  = "sparse2d",
        producer  = "sbndwire",
        name      = name+"data",
        MaxVoxels = 50000,
        Augment   = False,
        Channels  = [0,1,2]
    )

    # Bring in the labels:
    cb.add_batch_filler(
        datatype  = "sparse2d",
        producer  = "sbnd_cosmicseg",
        name      = name+"label",
        MaxVoxels = 50000,
        Augment   = False,
        Channels  = [0,1,2]
    )


    # Build up the data_keys:
    data_keys = {
        'image': name + 'data',
        'label': name + 'label',
        
        }

    if event_id or vertex_depth is not None:
        # Event-wide labels:
        cb.add_batch_filler(
            datatype = "particle",
            producer = "sbndneutrino",
            name     = name+"particle"
        )
        data_keys.update({'particle': name + 'particle'})


    if vertex_depth is not None:
        # Vertex locations as BBoxes:
        cb.add_batch_filler(
            datatype = "bbox2d",
            producer = "bbox_neutrino",
            name     = name + "vertex",
            MaxBoxes = 1,
            Channels = [0,1,2]
        )
        data_keys.update({'vertex': name + 'vertex'})




    # Prepare data managers:
    io_config = {
        'filler_name' : name,
        'filler_cfg'  : cb.get_config(),
        'verbosity'   : 5,
        'make_copy'   : False
    }

    return io_config, data_keys


def prepare_interface(batch_size, storage_name, larcv_interface, io_config, data_keys, color=None):

    """
    Not a pure function!  it changes state of the larcv_interface
    """
    larcv_interface.prepare_manager(
        storage_name, io_config, batch_size, data_keys, color=color)
    # This queues up the next data
    # self._larcv_interface.prepare_next(name)

    while larcv_interface.is_reading(storage_name):
        time.sleep(0.01)


    return larcv_interface.size(storage_name)


class synthetic_dataset(object):
    """Purely synthetic dataset 
    """

    def __init__(self, data_args, batch_size, event_id = False, vertex_depth=None):

        # Get the meta:
        dataset_version = data_args.version
        this_meta = meta(dataset_version)

        # To constuct the synthetic data, first determine the size:
        shape = batch_dims(
            data_args.data_format, 
            this_meta, 
            data_args.downsample, 
            batch_size)
        self.b_shape = shape
        self.i_shape = image_shape(this_meta, data_args.downsample)

        self.synthetic_index = 0
        self.batch_size = batch_size
        n_images = numpy.max([batch_size, 12])
        shape[0] = n_images

        self.init_data(shape)


    def init_data(self, shape):

        rng = numpy.random.default_rng()
        self.synthetic_images = rng.random(shape, dtype = numpy.float32)
        self.synthetic_labels = rng.integers(low=0, high=3, size=shape)


        # Should this include vertex and event id?  Yes if meta version is 2!
        # But also sometimes no, even in V2.0.  So flags are included.
        # ///TODO

    def __iter__(self):
        while True:
            yield self.get_batch()

    def __len__(self):
        return self.synthetic_labels.shape[0]

    def image_shape(self): return i_shape

    def batch_shape(self): return b_shape

    def get_batch(self):
        minibatch_data = {}
        if self.synthetic_index + self.batch_size > len(self.synthetic_images):
            self.synthetic_index = 0

        lower_index = self.synthetic_index
        upper_index = self.synthetic_index + self.batch_size

        minibatch_data['image']  = self.synthetic_images[lower_index:upper_index]
        minibatch_data['label']  = self.synthetic_labels[lower_index:upper_index]

        self.synthetic_index += 1

        return minibatch_data

def create_larcv_dataset(data_args, batch_size, input_file, name,
                         distributed=False, event_id=False, 
                         vertex_depth = None, sparse=False):
    """
    Create a new iterable dataset of the file specified in data_args
    pass

    """

    # To create a synthetic dataset:
    if data_args.synthetic:
        return synthetic_dataset(data_args, batch_size, event_id, vertex_depth)
    else:
        # Create a larcv interface:
        interface = create_larcv_interface(
            random_access_mode = data_args.random_mode, 
            distributed = distributed,
            seed=data_args.seed)

        # name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        # name = data_args.__class__.__name__

        # Next, prepare the config info for this interface:
        io_config, data_keys =  prepare_cosmic_tagger_config(
            batch_size = batch_size, 
            data_args = data_args, 
            input_file = input_file,
            name = name,
            event_id = event_id, 
            vertex_depth = vertex_depth)

        # Now, fire up the interface:
        prepare_interface(
            batch_size,
            storage_name = name, 
            larcv_interface = interface,
            io_config = io_config,
            data_keys = data_keys)

        shape = image_shape(meta(data_args.version), 2**data_args.downsample)

        # Finally, create the iterable object to hold all of this:
        dataset = larcv_dataset(
            larcv_interface = interface,
            shape           = shape,
            name            = name,
            data_args       = data_args, 
            event_id        = event_id,
            vertex_depth    = vertex_depth,
            sparse          = sparse)


        return dataset

class larcv_dataset(object):
    """ Represents a (possibly distributed) larcv dataset on one file

    Implements __len__ and __iter__ to enable fast, iterable datasets.

    May also in the future implement __getitem__(idx) to enable slower random access.

    """

    def __init__(self, larcv_interface, shape, name, data_args, event_id=False, vertex_depth=None, sparse=False):
        """
        Init takes a preconfigured larcv queue interface
        """

        self.larcv_interface = larcv_interface
        self.data_args       = data_args
        self.shape           = shape
        self.storage_name    = name
        self.vertex_depth    = vertex_depth
        self.event_id        = event_id  
        self.sparse          = sparse

        # self.data_keys = data_keys

        # Get image meta:
        self.image_meta = meta(data_args.version)

        self.stop = False

    def __len__(self):
        return self.larcv_interface.size(self.storage_name)


    def __iter__(self):
        
        while True:
            batch = self.fetch_next_batch(self.storage_name, force_pop=True)
            yield batch

            if self.stop:
                break

    def __del__(self):
        self.stop = True

    def image_size(self):
        return self.shape

    def batch_shape(seld):
        return batch_dims(self.data_args.data_format, self.image_meta, downsample, batch_size)


    def fetch_next_batch(self, name, force_pop=False):

        metadata=True

        pop = True
        if not force_pop:
            pop = False


        minibatch_data = self.larcv_interface.fetch_minibatch_data(self.storage_name,
            pop=pop,fetch_meta_data=metadata)
        minibatch_dims = self.larcv_interface.fetch_minibatch_dims(self.storage_name)


        # If the returned data is None, return none and don't load more:
        if minibatch_data is None:
            return minibatch_data

        # This brings up the next data to current data
        if pop:
            self.larcv_interface.prepare_next(self.storage_name)

        for key in minibatch_data:
            if key == 'entries' or key == 'event_ids':
                continue
            minibatch_data[key] = numpy.reshape(minibatch_data[key], minibatch_dims[key])

        n_neutrino_pixels = numpy.sum(minibatch_data['label'] == 2, axis=(1,2,3))

        # We need the event id for vertex classification, even if it's not used.
        if self.event_id or self.vertex_depth is not None:

            minibatch_data['event_label'] = data_transforms.event_label(
                minibatch_data['particle'][:,0],
                n_neutrino_pixels,
            )

        if self.vertex_depth is not None:
            downsample_level = 2**self.data_args.downsample

            # Put together the YOLO labels:
            minibatch_data["vertex"]  = data_transforms.form_yolo_targets(self.vertex_depth,
                minibatch_data["vertex"], minibatch_data["particle"],
                minibatch_data["event_label"], 
                self.data_args.data_format,
                self.image_meta, 
                downsample_level)


        # Get rid of the particle data now, we're done with it:
        if self.event_id or self.vertex_depth is not None:
            minibatch_data.pop("particle")




        if not self.sparse:
            minibatch_data['image']  = data_transforms.larcvsparse_to_dense_2d(
                minibatch_data['image'],
                dense_shape =self.shape,
                dataformat  =self.data_args.data_format)
        else:
            minibatch_data['image']  = data_transforms.larcvsparse_to_scnsparse_2d(
                minibatch_data['image'])

        # Label is always dense:
        minibatch_data['label']  = data_transforms.larcvsparse_to_dense_2d(
            minibatch_data['label'],
            dense_shape =self.shape,
            dataformat  =self.data_args.data_format)


        return minibatch_data
