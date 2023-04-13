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


def main():

    ####################################################################################
    # Definition of run parameters:
    NRANKS=984
    LOCAL_BATCH_SIZE=12
    FRAMEWORK="tensorflow"
    PRECISION="float32"
    DATA_FORMAT="channels_last"
    PREFIX="/data/datasets/Sunspot/ct_output_multirun/"
    RUN_NAME="synthetic-tile-run3"
    TITLE=f"{FRAMEWORK}-n{NRANKS}-df{DATA_FORMAT}-p{PRECISION}-mb{LOCAL_BATCH_SIZE}-{RUN_NAME}"
    top_dir = f"{PREFIX}/sunspot-a21-single-tile-{TITLE}/"
    OUT_DIR="test/"
    print(top_dir)
    ####################################################################################

    p = pathlib.Path(OUT_DIR)
    p.mkdir(parents=True, exist_ok=True)

    # Make a list of the GPUS:
    GPUS = [ f"{g//2}.{g%2}" for g in range(12) ]
    print(f"GPUS are {GPUS}")

    CPUS = list(range(0,45,8)) + list(range(52,96,8))
    print(f"CPUs are {CPUS}")


    hosts = get_hosts(top_dir)

    # print(f"Found the following hosts: {hosts}")

    df = create_dataframe(LOCAL_BATCH_SIZE, hosts, GPUS, CPUS, NRANKS, top_dir)

    bins = numpy.arange(15,21,0.07)



    plot_gpu_gpu_variation_scatter(df, GPUS, title=TITLE, output_dir=OUT_DIR)
    plot_gpu_gpu_variation_box(df, GPUS, title=TITLE, output_dir=OUT_DIR)
    plot_host_variation(df, title=TITLE, output_dir=OUT_DIR)
    histogram_tile_throughput(df, bins, title=TITLE, output_dir=OUT_DIR)
    plot_tile_FOM(df, bins, title=TITLE, output_dir=OUT_DIR)


def plot_gpu_gpu_variation_box(df, gpus, title, output_dir):

    fig = plt.figure(figsize=(16,9))


    for i, gpu in enumerate(gpus):
        sub_df = df.query(f"i_GPU == {i}")
        throughputs = sub_df['Throughputs'].values
        throughputs = numpy.concatenate(throughputs)
        
        
        plt.boxplot(throughputs, positions=(i,), showfliers=False)

    plt.xticks(range(12), gpus)
    plt.xlabel("GPU")
    plt.ylabel("Throughput [Img/s]")
    plt.grid(True)
    plt.title(title)
    plt.savefig(output_dir + "gpu_gpu_box.pdf")
    # plt.show()

def plot_gpu_gpu_variation_scatter(df, gpus, title, output_dir):

    fig = plt.figure(figsize=(16,9))

    for i, gpu in enumerate(gpus):
        # Filter for a particular GPU
        sub_df = df.query(f"i_GPU == {i}")
        
        plt.scatter(sub_df['GPU'], sub_df["Throughput"])
        

    plt.xlabel("GPU")
    plt.ylabel("Throughput [Img/s]")
    plt.grid(True)
    plt.title(title)
    
    plt.savefig(output_dir + "gpu_gpu_scatter.pdf")
    # plt.show()

def plot_host_variation(df, title, output_dir):


    fig = plt.figure(figsize=(16,9))
    # ax = fig.add_subplot(projection='3d')

    plt.scatter(df['i_Host'], df['Throughput'], marker='o', color='black')

    plt.xlabel("Node")
    plt.ylabel("Throughput [Img/s]")
    plt.grid(True)
    plt.title(title)
    plt.savefig(output_dir + "host_variation.pdf")
    
    # plt.show()

def histogram_tile_throughput(df, bins, title, output_dir):

    # Make a histogram of the iteration times

    fig = plt.figure(figsize=(16,9))


    throughputs_t1 = numpy.concatenate(df.query("tile")['Throughputs'].values)
    throughputs_t2 = numpy.concatenate(df.query("tile == False")['Throughputs'].values)
    #     print(throughputs)
    # counts, bin_edges = numpy.histogram(throughputs, bins=bins)

    counts_t1, bin_edges = numpy.histogram(throughputs_t1, bins=bins)
    counts_t2, bin_edges = numpy.histogram(throughputs_t2, bins=bins)

    mean_t1 = numpy.mean(df.query("tile")["Throughput"])
    mean_t2 = numpy.mean(df.query("tile == False")["Throughput"])


    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
    widths = bin_edges[1:] - bin_edges[:-1]
    plt.bar(bin_centers, counts_t1, width=widths, zorder=3, label=f"Tile 1, mean {mean_t1:.2f}", alpha=0.5)
    plt.bar(bin_centers, counts_t2, width=widths, zorder=3, label=f"Tile 2, mean {mean_t2:.2f}", alpha=0.5)
    plt.grid(zorder=0)
    plt.legend()
    plt.xlabel("Throughput [Img/s]")
    plt.gca().tick_params(labelleft=False) 
    plt.title(title)
    plt.savefig(output_dir + "tile_throughput.pdf")
    
    # plt.show()

def plot_tile_FOM(df, bins, title, output_dir):


    norm = scipy.stats.norm

    fom_per_tile = df['Throughput'].values

    fit = norm.fit(fom_per_tile)

    fig = plt.figure(figsize=(16,9))

    counts, bin_edges = numpy.histogram(fom_per_tile, bins=bins)
    min_fom = numpy.min(fom_per_tile)
    max_fom = numpy.max(fom_per_tile)

    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
    widths = bin_edges[1:] - bin_edges[:-1]

    # print(len(bins))
    # print(numpy.sum(bins))
    # print(widths)

    scale = 1000*widths[0]
    # print(scale)


    x = numpy.arange(numpy.min(bins),numpy.max(bins),0.1*widths[0])

    mean_fom = numpy.mean(fom_per_tile)


    plt.bar(bin_centers, counts, width=widths, zorder=3, label=f"Min/max {min_fom:.2f}/{max_fom:.2f} ({(max_fom -min_fom)/max_fom:.2f}%)")
    plt.plot(x, scale*norm(*fit).pdf(x), zorder=4, color="red", label=f"Gaussian ({fit[0]:.2f}, {fit[1]:.2f})")
    plt.grid(zorder=0)
    plt.legend()
    plt.xlabel("Throughput [Img/s]")
    plt.gca().tick_params(labelleft=False) 
    plt.title(title)
    plt.savefig(output_dir + "tile_FOM.pdf")
    # plt.show()


if __name__ == "__main__":
    main()