from . import larcv_io



# Here, we set up a bunch of template IO formats in the form of callable functions:

# These are all doing sparse IO, so there is no dense IO template here.  But you could add it.

def dataset_io(input_file, name, RandomAccess=None):
    data_proc  = gen_sparse2d_data_filler(name=name + "data",  producer="sbndwire")
    label_proc = gen_sparse2d_data_filler(name=name + "label", producer="sbnd_cosmicseg")

    config = larcv_io.ThreadIOConfig(name=name)

    config.add_process(data_proc)
    config.add_process(label_proc)

    config.set_param("InputFiles", input_file)
    if RandomAccess is not None:
        config.set_param("RandomAccess", RandomAccess)

    return config


def ana_io(input_file, name="", ):
    data_proc  = gen_sparse2d_data_filler(name=name + "data",  producer="sbndwire")
    label_proc = gen_sparse2d_data_filler(name=name + "label", producer="sbnd_cosmicseg")

    config = larcv_io.ThreadIOConfig(name="AnaIO")

    config.add_process(data_proc)
    config.add_process(label_proc)

    config.set_param("InputFiles", input_file)

    return config

def output_io(input_file):

    config = larcv_io.IOManagerConfig()

    config.set_param("InputFiles", input_file)

    return config

def gen_sparse2d_data_filler(name, producer):

    proc = larcv_io.ProcessConfig(proc_name=name, proc_type="BatchFillerSparseTensor2D")

    proc.set_param("Verbosity",         "3")
    proc.set_param("Tensor2DProducer",  producer)
    proc.set_param("IncludeValues",     "true")
    proc.set_param("MaxVoxels",         50000)
    proc.set_param("Channels",          "[0,1,2]")
    proc.set_param("UnfilledVoxelValue","-999")
    proc.set_param("Augment",           "false")

    return proc
