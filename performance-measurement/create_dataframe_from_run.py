import pathlib
import matplotlib
from matplotlib import pyplot as plt
import numpy
import scipy

import glob
import os

from datetime import datetime
import pandas as pd

# Define plot formatting:
font = {
        'family' : 'serif',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)
import matplotlib.font_manager

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", choices=["tf", "torch", "tensorflow", "pytorch"])
    parser.add_argument("-p", "--path", help = "Top directory of runs.")

    args = parser.parse_args()

    print(args)

# Here's a function to read the profiling data for a given run:
def read_numpy_profile_data(node, GPU, prefix, rank=0):
    data_file_name = prefix + "/" + node
    if GPU is not None:
        data_file_name += f"/GPU{GPU}" 
    data_file_name += f"/profiles/profiling_info_rank_{rank}.npy"
    # print(data_file_name)
    read_in_data = numpy.load(data_file_name)

    return read_in_data

def get_hosts(top_dir):
    hosts = glob.glob(top_dir + "/x*")
    hosts = [os.path.basename(h) for h in hosts]
    return hosts

def validate_run_success(hosts, gpus, prefix):
    '''
    Return the success/failure of jobs based on whether or not the final profiling
    data was created.  We test if it was created by reading it in.

    Return is a dictionary, perhaps nested by GPU identifier, of True/False
    meaning pass/fail.
    '''
    run_results = {}
    print(prefix)
    
    for i_host, host in enumerate(hosts):
        if gpus is not None:
        
            if host not in run_results: run_results[host] = {}

            for i_gpu, GPU in enumerate(gpus):
                try:
                    this_data = read_numpy_profile_data(host, GPU, prefix)
                    run_results[host][GPU] = True
                except:
                    run_results[host][GPU] = False
                    print(f"Host {host} and GPU {GPU} FAILED")
                    continue
        else:
            try:
                this_data = read_numpy_profile_data(host, GPU=None, prefix=prefix)
                run_results[host] = True
            except:
                run_results[host] = False
                print(f"Host {host} and GPU {GPU} FAILED")
                continue

    return run_results
    
def create_dataframe_single_tile(local_batch_size, hosts, gpus, nranks, prefix):
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
                this_data = read_numpy_profile_data(host, GPU, prefix)
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


def create_dataframe_single_node(local_batch_size, hosts, nranks, prefix):
    host_vals  = []
    host_index = []
    throughputs = []
    variation  = []
    start_time = []
    all_imgs   = []

    for i_host, host in enumerate(hosts):
        # try:
        this_data = read_numpy_profile_data(host, GPU=None, prefix=prefix)
        # except:
        # print(f"Host {host} and GPU {GPU} FAILED")
        # continue

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
        host_vals.append(host)
        throughputs.append(average_img_per_s)
        variation.append(var_img_per_s)
        all_imgs.append(img_per_s)

    # throughput = len(data['iteration'][2:]) * LOCAL_BATCH_SIZE / numpy.sum(numpy.cast(data['iteration'][2:].cast(numpy.float))
    # print(throughput)

    df = pd.DataFrame(
        zip(host_index, host_vals,
            throughputs, all_imgs, variation, start_time),
        columns=["i_Host", "Host",
                 "Throughput", "Throughputs",
                 "Uncert", "Start"])



    return df

def create_dataframe_multi_node(local_batch_size, nranks, prefix):
    throughputs = []
    variation  = []
    start_time = []
    all_imgs   = []
    ranks      = []

    for rank in range(nranks):
        this_data = read_numpy_profile_data(node="", GPU=None, prefix=prefix, rank=rank)

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


        throughputs.append(average_img_per_s)
        variation.append(var_img_per_s)
        all_imgs.append(img_per_s)
        ranks.append(rank)

    # throughput = len(data['iteration'][2:]) * LOCAL_BATCH_SIZE / numpy.sum(numpy.cast(data['iteration'][2:].cast(numpy.float))
    # print(throughput)
    df = pd.DataFrame(
        zip(ranks, throughputs, all_imgs, variation, start_time),
        columns=["Rank",
                 "Throughput", "Throughputs",
                 "Uncert", "Start"])



    return df



if __name__ == '__main__':
    main()
