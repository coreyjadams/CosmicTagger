import os
import time

from . import data_transforms
import tempfile

import numpy

import logging
logger = logging.getLogger("cosmictagger")

from src.config import DataFormatKind


class larcv_fetcher(object):


    # This represents the intrinsic image shape:
    _image_meta1 = numpy.array([
        ([1280, 2048], [1280, 2048],[0.,0.]),
        ([1280, 2048], [1280, 2048],[0.,0.]),
        ([1280, 2048], [1280, 2048],[0.,0.])],
        dtype=[
            ('full_pixels', "int", (2)),
            ('size', "float", (2)),
            ('origin', "float", (2)),
        ]
    )

    _image_meta2 = numpy.array([
        ([1408, 2048], [439.4660061919505, 614.4],[-18.1030030959, -9.6]),
        ([1408, 2048], [439.4660061919505, 614.4],[-18.1030030959, -9.6]),
        ([1408, 2048], [439.4660061919505, 614.4],[-18.1030030959, -57.5999999])],
        dtype=[
            ('full_pixels', "int", (2)),
            ('size', "float", (2)),
            ('origin', "float", (2)),
        ]
    )


    def __init__(self, mode, distributed, data_args, sparse, seed=None, vtx_depth=None):

        if mode not in ['train', 'inference', 'iotest']:
            raise Exception("Larcv Fetcher can't handle mode ", mode)



        self.data_args = data_args
        self.mode       = mode
        self.sparse     = sparse
        self.vtx_depth  = vtx_depth
        self.writer     = None

        if not self.data_args.synthetic:

            if distributed:
                from larcv import distributed_queue_interface as queueloader
            else:
                from larcv import queueloader

            if mode == "inference":
                self._larcv_interface = queueloader.queue_interface(
                    random_access_mode="serial_access", seed=0)
                self._larcv_interface.no_warnings()
            elif mode == "train" or mode == "iotest":
                self._larcv_interface = queueloader.queue_interface(
                    random_access_mode="random_blocks", seed=seed)
                self._larcv_interface.no_warnings()
            else:
                # Must be synthetic
                self._larcv_interface = None


        # Compute the realized image shape:
        self.ds = 2**self.data_args.downsample

        if self.data_args.version == 1:
            self.image_meta = self._image_meta1
        else:
            self.image_meta = self._image_meta2

        self.image_shape = [ int(i / self.ds) for i in self.image_meta['full_pixels'][0] ]


    def __del__(self):
        if hasattr(self, "writer") and self.writer is not None:
            self.writer.finalize()


    def image_size(self):
        '''Return the input shape to the networks (no batch size)'''
        return self.image_shape

    def batch_dims(self, batch_size):

        if self.data_args.data_format == DataFormatKind.channels_first:
            shape = [batch_size, 3, self.image_shape[0], self.image_shape[1]]
        elif self.data_args.data_format == DataFormatKind.channels_last:
            shape = [batch_size, self.image_shape[0], self.image_shape[1], 3]
        else:
            raise Exception(f"Unknown data format: {args.data.data_format}")
        return shape

    def prepare_cosmic_sample(self, name, input_file, batch_size, color=None):


        if self.data_args.synthetic:
            self.synthetic_index = 0
            self.batch_size = batch_size
            n_images = numpy.max([batch_size, 12])
            shape = self.batch_dims(n_images)

            self.synthetic_images = numpy.random.random_sample(shape).astype(numpy.float32)
            self.synthetic_labels = numpy.random.randint(low=0, high=3, size=shape)

            return n_images

        else:

            # First, verify the files exist:
            if not os.path.exists(input_file):
                raise Exception(f"File {input_file} not found")


            from larcv.config_builder import ConfigBuilder
            cb = ConfigBuilder()
            cb.set_parameter([str(input_file)], "InputFiles")
            cb.set_parameter(5, "ProcessDriver", "IOManager", "Verbosity")
            cb.set_parameter(5, "ProcessDriver", "Verbosity")
            cb.set_parameter(5, "Verbosity")

            # Do we need to do compression?
            if self.data_args.downsample != 0:
                cb.add_preprocess(
                    datatype = "sparse2d",
                    Product  = "sparse2d",
                    producer = "sbndwire",
                    process  = "Downsample",
                    OutputProducer = "sbndwire",
                    Downsample = self.ds,
                    PoolType = 1 # average,
                )
                cb.add_preprocess(
                    datatype = "sparse2d",
                    Product  = "sparse2d",
                    producer = "sbnd_cosmicseg",
                    process  = "Downsample",
                    OutputProducer = "sbnd_cosmicseg",
                    Downsample = self.ds,
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

            # Event-wide labels:
            cb.add_batch_filler(
                datatype = "particle",
                producer = "sbndneutrino",
                name     = name+"particle"
            )


            # Vertex locations as BBoxes:
            cb.add_batch_filler(
                datatype = "bbox2d",
                producer = "bbox_neutrino",
                name     = name + "vertex",
                MaxBoxes = 1,
                Channels = [0,1,2]
            )


            # Build up the data_keys:
            data_keys = {
                'image': name + 'data',
                'label': name + 'label',
                'particle': name + 'particle',
                'vertex': name + 'vertex'
                }


            # Prepare data managers:
            io_config = {
                'filler_name' : name,
                'filler_cfg'  : cb.get_config(),
                'verbosity'   : 5,
                'make_copy'   : False
            }

            self._larcv_interface.prepare_manager(name, io_config, batch_size, data_keys, color=color)

            #
            # if self.mode == "inference":
            #     self._larcv_interface.set_next_index(name, start_index)

            # This queues up the next data
            # self._larcv_interface.prepare_next(name)

            while self._larcv_interface.is_reading(name):
                time.sleep(0.01)

            logger.info("Larcv file prepared")

            return self._larcv_interface.size(name)


    def fetch_next_batch(self, name, force_pop=False):

        if not self.data_args.synthetic:
            metadata=True

            pop = True
            if not force_pop:
                pop = False


            minibatch_data = self._larcv_interface.fetch_minibatch_data(name,
                pop=pop,fetch_meta_data=metadata)
            minibatch_dims = self._larcv_interface.fetch_minibatch_dims(name)


            # If the returned data is None, return none and don't load more:
            if minibatch_data is None:
                return minibatch_data

            # This brings up the next data to current data
            if pop:
                self._larcv_interface.prepare_next(name)

            for key in minibatch_data:
                if key == 'entries' or key == 'event_ids':
                    continue
                minibatch_data[key] = numpy.reshape(minibatch_data[key], minibatch_dims[key])

            n_neutrino_pixels = numpy.sum(minibatch_data['label'] == 2, axis=(1,2,3))


            minibatch_data['event_label'] = data_transforms.event_label(
                minibatch_data['particle'][:,0],
                n_neutrino_pixels,
            )

            # Put together the YOLO labels:
            minibatch_data["vertex"]  = data_transforms.form_yolo_targets(self.vtx_depth,
                minibatch_data["vertex"], minibatch_data["particle"],
                minibatch_data["event_label"], self.data_args.data_format,
                self.image_meta, self.ds)

            # Get rid of the particle data now, we're done with it:
            minibatch_data.pop("particle")




            if not self.sparse:
                minibatch_data['image']  = data_transforms.larcvsparse_to_dense_2d(
                    minibatch_data['image'],
                    dense_shape =self.image_shape,
                    dataformat  =self.data_args.data_format)
            else:
                minibatch_data['image']  = data_transforms.larcvsparse_to_scnsparse_2d(
                    minibatch_data['image'])

            # Label is always dense:
            minibatch_data['label']  = data_transforms.larcvsparse_to_dense_2d(
                minibatch_data['label'],
                dense_shape =self.image_shape,
                dataformat  =self.data_args.data_format)

        else:

            minibatch_data = {}
            if self.synthetic_index + self.batch_size > len(self.synthetic_images):
                self.synthetic_index = 0

            lower_index = self.synthetic_index
            upper_index = self.synthetic_index + self.batch_size

            minibatch_data['image']  = self.synthetic_images[lower_index:upper_index]
            minibatch_data['label']  = self.synthetic_labels[lower_index:upper_index]

            self.synthetic_index += 1

        return minibatch_data

    def prepare_writer(self, input_file, output_file):

        from larcv import larcv_writer
        config = io_templates.output_io(input_file  = input_file)

        # Generate a named temp file:
        main_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        main_file.write(config.generate_config_str())

        main_file.close()

        self.writer = larcv_writer.larcv_writer(main_file.name, output_file)

    def write(self, data, producer, entry, event_id):
        self.writer.write(data, datatype='sparse2d', producer=producer, entry=entry, event_id=event_id)
