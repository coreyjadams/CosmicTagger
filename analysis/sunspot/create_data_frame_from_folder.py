import pathlib
import numpy

import glob
import os

import pandas as pd



# Here's a function to read the profiling data for a given run:
def read_numpy_profile_data(node, GPU, CPU, prefix):
    data_file_name = prefix + node + f"/GPU{GPU}-CPU{CPU}" + f"/profiles/profiling_info_rank_0.npy"
    # print(data_file_name)
    read_in_data = numpy.load(data_file_name)
    
    return read_in_data

def get_hosts(top_dir):
    hosts = glob.glob(top_dir + "/x*")
    hosts = [os.path.basename(h) for h in hosts]
    return hosts    


def create_dataframe(local_batch_size, hosts, gpus, cpus, nranks, prefix):
    host_vals  = []
    host_index = []
    gpu_index  = []
    gpu_vals   = []
    throughputs = []
    variation  = []
    start_time = []
    all_imgs   = []

    for i_host, host in enumerate(hosts):
        for i_gpu, GPU in enumerate(gpus):
            try:
                this_data = read_numpy_profile_data(host, GPU, cpus[i_gpu], prefix)
            except:
                print(f"Host {host} and GPU {GPU} FAILED")
                continue
                
            start_time.append( this_data[0]['start'] )
            # Compute img/s for each iteration:
            img_per_s = local_batch_size / (1e-6*this_data['iteration'][2:].astype(float) )
            
            # Compute throughput as total time / total iterations
            total_time = 1e-6*numpy.sum(this_data['iteration'][2:].astype(float))
            total_iterations = len(this_data['iteration'][2:])

            this_throughput = local_batch_size * total_iterations / total_time
                
            average_img_per_s = numpy.mean(img_per_s)
            var_img_per_s     = numpy.std(img_per_s)
    #         print(this_throughput)
    #         print(average_img_per_s)
            
            
            host_index.append(i_host)
            gpu_index.append(i_gpu)
            gpu_vals.append(GPU)
            host_vals.append(host)
            throughputs.append(average_img_per_s)
            variation.append(var_img_per_s)
            all_imgs.append(img_per_s)

    # throughput = len(data['iteration'][2:]) * LOCAL_BATCH_SIZE / numpy.sum(numpy.cast(data['iteration'][2:].cast(numpy.float))
    # print(throughput)

    df = pd.DataFrame(
        zip(host_index, host_vals, gpu_index, 
            gpu_vals, throughputs, all_imgs, variation, start_time), 
        columns=["i_Host", "Host", "i_GPU", 
                 "GPU", "Throughput", "Throughputs", 
                 "Uncert", "Start"])

    # Add a column for the tile:
    df['tile'] = df['i_GPU'] % 2 == 0


    return df
