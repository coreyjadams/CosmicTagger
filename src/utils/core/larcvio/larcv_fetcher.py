import os
import torch 

from . import data_transforms
from . import io_templates
import tempfile

import numpy

from skimage.measure import block_reduce

class larcv_fetcher(object):

    FULL_RESOLUTION_H = 1280
    FULL_RESOLUTION_W = 2048

    def __init__(self, mode, distributed, downsample, dataformat, synthetic, sparse, seed=None):

        if mode not in ['train', 'inference']:
            raise Exception("Larcv Fetcher can't handle mode ", mode)

        if distributed:
            from larcv import distributed_queue_interface 
            self._larcv_interface = distributed_queue_interface.queue_interface()
        else:
            from larcv import queueloader
            if mode == "inference":
                self._larcv_interface = queueloader.queue_interface(
                    random_access_mode="serial_access", seed=seed)
            elif mode == "train":
                self._larcv_interface = queueloader.queue_interface(
                    random_access_mode="random_blocks", seed=seed)
            else:
                # Must be synthetic
                self._larcv_interface = None

        self.mode       = mode
        self.downsample = downsample
        self.dataformat = dataformat
        self.synthetic  = synthetic
        self.sparse     = sparse
        
        # self._cleanup = []

        # Compute the realized image shape:
        self.full_image_shape = [self.FULL_RESOLUTION_H, self.FULL_RESOLUTION_W]
        self.ds = 2**downsample

        self.image_shape = [ int(i / self.ds) for i in self.full_image_shape ]

    # def __del__(self):
    #     for f in self._cleanup:
    #         os.unlink(f.name)

    def image_shape(self):
        '''Return the input shape to the networks (no batch size)'''
        return self.image_shape

    def batch_dims(self, batch_size):

        if self.dataformat == "channels_first":
            shape = [batch_size, 3, self.image_shape[0], self.image_shape[1]]
        else:
            shape = [batch_size, self.image_shape[0], self.image_shape[1], 3]

        return shape

    def prepare_cosmic_sample(self, name, input_file, batch_size, color=None):

        if self.synthetic:
            self.synthetic_index = 0
            self.batch_size = batch_size
            shape = self.batch_dims(12)

            self.synthetic_images = numpy.random.random_sample(shape)
            self.synthetic_weight = numpy.random.random_sample(shape)
            self.synthetic_labels = numpy.random.randint(low=0, high=3, size=shape)

        else:
            config = io_templates.dataset_io(
                    input_file = input_file,
                    name       = name)


            # Generate a named temp file:
            main_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
            main_file.write(config.generate_config_str())

            main_file.close()

            io_config = {
                'filler_name' : config._name,
                'filler_cfg'  : main_file.name,
                'verbosity'   : 5,
                'make_copy'   : False
            }

            # Build up the data_keys:
            data_keys = {
                'image': name + 'data',
                'label': name + 'label'
                }

            self._larcv_interface.prepare_manager(name, io_config, batch_size, data_keys, color=color)
            os.unlink(main_file.name)


            return self._larcv_interface.size(name)

    # def prepare_eventID_output(self, name, input_file, output_file):
    #     config = io_templates.output_io(input_file=input_file, output_file=output_file)
            
    #     out_file_config = tempfile.NamedTemporaryFile(mode='w', delete=False)
    #     out_file_config.write(config.generate_config_str())
    #     print(config.generate_config_str())

    #     out_file_config.close()
    #     self._cleanup.append(out_file_config)

    #     self._larcv_interface.prepare_writer(out_file.name, output_file)


    def fetch_next_batch(self, name, force_pop=False):


        if not self.synthetic:
            metadata=True

            pop = True
            if not force_pop:
                pop = False


            # This brings up the current data
            self._larcv_interface.prepare_next(name)
            minibatch_data = self._larcv_interface.fetch_minibatch_data(name, 
                pop=pop,fetch_meta_data=metadata)
            minibatch_dims = self._larcv_interface.fetch_minibatch_dims(name)


            for key in minibatch_data:
                if key == 'entries' or key == 'event_ids':
                    continue
                minibatch_data[key] = numpy.reshape(minibatch_data[key], minibatch_dims[key])


            if not self.sparse:
                minibatch_data['image']  = data_transforms.larcvsparse_to_dense_2d(
                    minibatch_data['image'], 
                    dense_shape =self.full_image_shape, 
                    dataformat  =self.dataformat)
            else:
                minibatch_data['image']  = data_transforms.larcvsparse_to_scnsparse_2d(
                    minibatch_data['image'])

            # Label is always dense:
            minibatch_data['label']  = data_transforms.larcvsparse_to_dense_2d(
                minibatch_data['label'], 
                dense_shape =self.full_image_shape, 
                dataformat  =self.dataformat)


            # Now, this is supposed to be temporary, but we need to downsample too:
            if self.downsample != 0:

                if self.dataformat == "channels_first":
                    kernel = (1, 1, self.ds, self.ds)
                else:
                    kernel = (1, self.ds, self.ds, 1)

                minibatch_data['image'] = block_reduce(
                    minibatch_data['image'],
                    kernel,
                    func = numpy.mean)

                minibatch_data['label'] = block_reduce(
                    minibatch_data['label'],
                    kernel,
                    func = numpy.max)
     
        else:

            minibatch_data = {}
            if self.synthetic_index + self.batch_size > len(self.synthetic_images):
                self.synthetic_index = 0

            lower_index = self.synthetic_index
            upper_index = self.synthetic_index + self.batch_size

            minibatch_data['image']  = self.synthetic_images[lower_index:upper_index]
            minibatch_data['label']  = self.synthetic_labels[lower_index:upper_index]
            minibatch_data['weight'] = self.synthetic_images[lower_index:upper_index]

            self.synthetic_index += 1

        return minibatch_data



    # def to_torch_cycleGAN(self, minibatch_data, device=None):

    #     if device is None:
    #         if torch.cuda.is_available():
    #             device = torch.device('cuda')
    #         else:
    #             device = torch.device('cpu')


    #     for key in minibatch_data:
    #         if key == 'entries' or key =='event_ids':
    #             continue
    #         else:
    #             minibatch_data[key] = torch.tensor(minibatch_data[key],device=device)
        
    #     return minibatch_data


    # def to_torch_eventID(self, minibatch_data, device=None):

    #     if device is None:
    #         if torch.cuda.is_available():
    #             device = torch.device('cuda')
    #         else:
    #             device = torch.device('cpu')


    #     for key in minibatch_data:
    #         if key == 'entries' or key =='event_ids':
    #             continue
    #         if key == 'image' and self.args.SPARSE:
    #             if self.args.INPUT_DIMENSION == '3D':
    #                 minibatch_data['image'] = (
    #                         torch.tensor(minibatch_data['image'][0]).long(),
    #                         torch.tensor(minibatch_data['image'][1], device=device),
    #                         minibatch_data['image'][2],
    #                     )
    #             else:
    #                 minibatch_data['image'] = (
    #                         torch.tensor(minibatch_data['image'][0]).long(),
    #                         torch.tensor(minibatch_data['image'][1], device=device),
    #                         minibatch_data['image'][2],
    #                     )
    #         else:
    #             minibatch_data[key] = torch.tensor(minibatch_data[key],device=device)
        
    #     return minibatch_data