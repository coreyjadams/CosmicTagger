from . import larcv_io



# Here, we set up a bunch of template IO formats in the form of callable functions:

# These are all doing sparse IO, so there is no dense IO template here.  But you could add it.

def train_io(input_file, max_voxels, data_producer, label_producer, prepend_names=""):
    data_proc = gen_sparse2d_data_filler(name=prepend_names + "data", producer=data_producer, max_voxels=max_voxels)
    label_proc = gen_sparse2d_data_filler(name=prepend_names + "label", producer=label_producer, max_voxels=max_voxels)


    config = larcv_io.ThreadIOConfig(name="TrainIO")

    config.add_process(data_proc)
    config.add_process(label_proc)

    config.set_param("InputFiles", input_file)

    return config


def test_io(input_file, max_voxels, data_producer, label_producer, prepend_names="aux_"):
    data_proc = gen_sparse2d_data_filler(name=prepend_names + "data", producer=data_producer, max_voxels=max_voxels)
    label_proc = gen_sparse2d_data_filler(name=prepend_names + "label", producer=label_producer, max_voxels=max_voxels)


    config = larcv_io.ThreadIOConfig(name="TestIO")

    config.add_process(data_proc)
    config.add_process(label_proc)

    config.set_param("InputFiles", input_file)

    return config


def ana_io(input_file, max_voxels, data_producer, label_producer, prepend_names=""):
    data_proc = gen_sparse2d_data_filler(name=prepend_names + "data", producer=data_producer, max_voxels=max_voxels)
    label_proc = gen_sparse2d_data_filler(name=prepend_names + "label", producer=label_producer, max_voxels=max_voxels)


    config = larcv_io.ThreadIOConfig(name="AnaIO")

    config.add_process(data_proc)
    config.add_process(label_proc)

    config.set_param("InputFiles", input_file)
    config.set_param("RandomAccess", 0)

    return config

def output_io(input_file):

    config = larcv_io.IOManagerConfig()

    config.set_param("InputFiles", input_file)

    return config

def gen_sparse2d_data_filler(name, producer, max_voxels):

    proc = larcv_io.ProcessConfig(proc_name=name, proc_type="BatchFillerSparseTensor2D")

    proc.set_param("Verbosity",         "3")
    proc.set_param("Tensor2DProducer",  producer)
    proc.set_param("IncludeValues",     "true")
    proc.set_param("MaxVoxels",         max_voxels)
    proc.set_param("Channels",          "[0,1,2]")
    proc.set_param("UnfilledVoxelValue","-999")
    proc.set_param("Augment",           "false")

    return proc

def gen_sparse3d_data_filler(name, producer, max_voxels):

    proc = larcv_io.ProcessConfig(proc_name=name, proc_type="BatchFillerSparseTensor3D")

    proc.set_param("Verbosity",         "3")
    proc.set_param("Tensor3DProducer",  producer)
    proc.set_param("IncludeValues",     "true")
    proc.set_param("MaxVoxels",         max_voxels)
    proc.set_param("UnfilledVoxelValue","-999")
    proc.set_param("Augment",           "true")

    return proc


def gen_label_filler(label_mode, prepend_names, n_classes):

    proc = larcv_io.ProcessConfig(proc_name=prepend_names + "label", proc_type="BatchFillerPIDLabel")

    proc.set_param("Verbosity",         "3")
    proc.set_param("ParticleProducer",  "label")
    proc.set_param("PdgClassList",      "[{}]".format(",".join([str(i) for i in range(n_classes)])))

    return proc
