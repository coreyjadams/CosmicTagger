from pathlib import Path
import matplotlib
from matplotlib import pyplot as plt
import numpy
import scipy

import glob
import os, shutil

from datetime import datetime
import pandas as pd

# Define plot formatting:
font = {
        'family' : 'serif',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)
import matplotlib.font_manager

from common import get_parser, Framework, System, make_run_id, get_affinity
from create_dataframe_from_run import create_dataframe_single_tile
from create_dataframe_from_run import create_dataframe_single_node
from create_dataframe_from_run import create_dataframe_multi_node

bounds = {
    "tf" : [16, 25],
    "pt" : [20, 30],
}
bins = {
    key : numpy.arange(bounds[key][0], bounds[key][1], 0.1) for key in bounds.keys()
}


def infer_hosts(top_dir):

    # There should be a hosts file in the top dir:
    host_file = top_dir / Path("hosts/hosts.txt")
    with open(host_file, 'r') as f:
        hosts = f.readlines()

    hosts = [ h.rstrip("\n") for h in hosts ]
    hosts = [ h.split(".")[0] for h in hosts]

    # Ensure the hosts come back in a specific order:
    return hosts

def main(args):

    OUT_DIR= args.out_dir / Path("./analysis/")
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    hosts = infer_hosts(args.out_dir)

    host_file = args.out_dir / Path("hosts/hosts.txt")
    host_dest = OUT_DIR / Path("hosts.txt")
    # Copy the host file to the analysis dir:
    shutil.copyfile(host_file, host_dest)


    # store the performance by framework, it's a dataframe per framework
    single_tile_performance = {}
    single_node_performance = {}
    scale_out_performance   = {}

    affinity = get_affinity(args.system)

    # print(affinity["gpu_affinity"])


        

    TITLE=""

    # Read in the data from the single-tile jobs:
    for framework in [Framework.tf, Framework.pt]:

        run_id = make_run_id(args, framework,  hosts)
        this_path = args.out_dir / Path("single-tile/" + run_id)

        
        this_df = create_dataframe_single_tile(
            local_batch_size = args.batch_size, 
            hosts = hosts, 
            gpus = affinity["gpu_affinity"], 
            nranks = 12, 
            prefix = str(this_path)
        )

        this_out_dir = OUT_DIR / Path("single-tile/" + run_id + "/")
        this_out_dir.mkdir(exist_ok=True, parents=True)


        csv_path = this_out_dir / Path("raw_data.csv") 

        this_df.to_csv(csv_path)

        single_tile_performance[framework.name] = this_df

        titlebase = f"{args.system.name}-{framework.name}-MB{args.batch_size}"
        plot_gpu_gpu_variation_scatter(this_df, affinity["gpu_affinity"],
            title=titlebase+" GPU Variation", output_dir=this_out_dir)

        # plot_gpu_gpu_variation_box(this_df, affinity["gpu_affinity"], 
        #                            title=titlebase + " GPU Variation", output_dir=this_out_dir)
        plot_host_variation(this_df, title=titlebase + " Host Variation", output_dir=this_out_dir)
        
        
        histogram_tile_throughput(this_df, bins[framework.name], 
                                  title=titlebase + " Tile Throughput", output_dir=this_out_dir)
        plot_tile_FOM(this_df, bins[framework.name], 
                      title=titlebase + " Aurora FOM", output_dir=this_out_dir)

    # Here, make a scatter plot of tf vs hvd performance on the same tile:
    this_out_dir = OUT_DIR / Path("single-tile/comparison/")
    this_out_dir.mkdir(exist_ok=True, parents=True)
    
    compare_tile_perf(single_tile_performance, title="FOM, TF vs. PT per Tile", output_dir=this_out_dir)


    for framework in [Framework.tfhvd, Framework.ptddp, Framework.pthvd]:
        run_id = make_run_id(args, framework,  hosts)

        this_path = args.out_dir / Path("single-node/" + run_id)

        nranks = len(affinity["gpu_affinity"])

        single_node_performance[framework.name] = create_dataframe_single_node(
            local_batch_size = args.batch_size, 
            hosts = hosts, 
            nranks = nranks, 
            prefix = str(this_path)
        )

        this_out_dir = OUT_DIR / Path("single-node/" + run_id + "/")
        this_out_dir.mkdir(exist_ok=True, parents=True)

        csv_path = this_out_dir / Path("raw_data.csv") 

        single_node_performance[framework.name].to_csv(csv_path)


        # print(single_node_performance[framework.name])

        single_tile_framework = Framework.tf if "tf" in framework.name else Framework.pt
        
        titlebase = f"{args.system.name}-{framework.name}-MB{args.batch_size}"

        plot_scale_up_by_host(
            single_node_performance[framework.name], 
            single_tile_performance[single_tile_framework.name],
            gpus = affinity["gpu_affinity"],
            title=titlebase + " Scale UP by Host",
            output_dir=this_out_dir)


    # And now, multi-node performance:
    for framework in [Framework.tfhvd, Framework.ptddp, Framework.pthvd]:
        run_id = make_run_id(args, framework,  hosts)

        this_path = args.out_dir / Path("multi-node/" + run_id)

        nranks = len(hosts) * len(affinity["gpu_affinity"])

        scale_out_performance[framework.name] = create_dataframe_multi_node(
            local_batch_size = args.batch_size, 
            nranks = nranks, 
            prefix = str(this_path)
        )


        this_out_dir = OUT_DIR / Path("multi-node/" + run_id + "/")
        this_out_dir.mkdir(exist_ok=True, parents=True)

        csv_path = this_out_dir / Path("raw_data.csv") 

        scale_out_performance[framework.name].to_csv(csv_path)
        titlebase = f"{args.system.name}-{framework.name}-MB{args.batch_size}"

        single_tile_framework = Framework.tf if "tf" in framework.name else Framework.pt
        plot_scale_out(
            scale_out_performance[framework.name],
            single_node_performance[framework.name], 
            single_tile_performance[single_tile_framework.name],
            bins = bins[single_tile_framework.name],
            title=titlebase + " Scale Out Gaps",
            output_dir=this_out_dir)

def compare_tile_perf(perf_dict, title, output_dir):

    fig = plt.figure(figsize=(16,16))

    t1_tf_throughput = perf_dict['tf'].query("tile")['Throughput']
    t2_tf_throughput = perf_dict['tf'].query("tile==False")['Throughput']
    t1_pt_throughput = perf_dict['pt'].query("tile")['Throughput']
    t2_pt_throughput = perf_dict['pt'].query("tile==False")['Throughput']

    # Tile 1
    plt.scatter(t1_tf_throughput, t1_pt_throughput, label="Tile 1")
    plt.scatter(t2_tf_throughput, t2_pt_throughput, label="Tile 2")

    # plt.ylim([T_MIN, T_MAX])
    # plt.xlim([T_MIN, T_MAX])
    plt.xlabel("Tensorflow [Img/s]")
    plt.ylabel("Pytorch [Img/s]")
    plt.grid(True)
    plt.title(title)
    plt.legend()
    plt.savefig(str(output_dir) + "/tile_scatter.pdf")


def plot_scale_out(scale_out_df, scale_up_df, single_tile_df, bins, title, output_dir):

    fig = plt.figure(figsize=(16,9))

    # How to plot the scale out efficiency?
    # A 1D histogram with efficiency drops included seems most reasonable

    # First, add the single-tile measurements:


    fom_per_tile = single_tile_df['Throughput'].values

    counts_tile, bin_edges = numpy.histogram(fom_per_tile, bins=bins)
    min_fom_tile = numpy.min(fom_per_tile)
    mean_fom_tile = numpy.mean(fom_per_tile)

    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
    widths = bin_edges[1:] - bin_edges[:-1]


    # Normalize to peak at 1.0:
    scale_tile = numpy.max(counts_tile)
    counts_tile = counts_tile / scale_tile


    plt.bar(bin_centers, counts_tile, width=widths, zorder=3, 
        label=f"Single Tile (Min/Mean: {min_fom_tile:.2f}/{mean_fom_tile:.2f})")
    
    # Plot the single-node values:
    fom_per_node = scale_up_df["Throughput"].values
    counts_node, bin_edges = numpy.histogram(fom_per_node, bins=bins)
    min_fom_node = numpy.min(fom_per_node)
    mean_fom_node = numpy.mean(fom_per_node)
    
    # Normalize to peak at 1.0:
    scale_node = numpy.max(counts_node)
    counts_node = counts_node / scale_node

    plt.bar(bin_centers, counts_node, width=widths, zorder=3, 
        label=f"Single Node (Min/Mean: {min_fom_node:.2f}/{mean_fom_node:.2f})")

    fom_scale_out = numpy.mean(scale_out_df["Throughput"].values)

    n_ranks = scale_out_df['Rank'].max() + 1
    
    # Break down the in-efficiencies.
    # We get some inefficiency from hardware variations at the tile level:
    # The "ideal" FOM is the mean of the tile_fom.
    # The hardware inefficiency is the gap from mean to min fom:
    hardware_gap = (mean_fom_tile - min_fom_tile) / mean_fom_tile

    # The scale-up gap is the worst node-level efficiency, so the fraction 
    # of ideal fom lost comparing worst node to worst tile:
    scale_up_gap = (min_fom_tile - min_fom_node  ) / mean_fom_tile

    # Overall efficiency is the global fom over mean fom:
    overall_gap  = fom_scale_out / mean_fom_tile
    scale_out_gap = 1 - overall_gap - scale_up_gap - hardware_gap
    # scale_out_gap = 1 - scale_out_gap


    plt.bar(fom_scale_out, 1.0, width=widths[0], zorder=3, 
        label=f"Scale Out {n_ranks} ranks, {100*overall_gap:.2f}%")

    # Add shading for hardware vs. communication inefficiencies:
    # Communication:
    plt.gca().axvspan(fom_scale_out, min_fom_node, alpha=0.25, color='red', zorder=3,
                      ymin=0.0, ymax=0.5,
                      )
    plt.text(x=0.5*(fom_scale_out+min_fom_node), 
             y = 1.03, s=f"-{100*scale_out_gap:.2f}%s",
             ha="center", rotation="vertical")

    plt.gca().axvspan(min_fom_tile, numpy.mean(fom_per_tile), alpha=0.25, color='orange', zorder=3,
                      ymin=0.0, ymax=0.5,
                      )
    plt.text(x=0.5*(min_fom_tile + numpy.mean(fom_per_tile)), 
             y = 1.03, 
             s=f"-{100*hardware_gap:.2f}%s", 
             ha="center", rotation="vertical")

    plt.gca().axvspan(min_fom_node, min_fom_tile, alpha=0.25, color='purple', zorder=3,
                      ymin=0.0, ymax=0.5,
                      
                    #   label=f"Hardware variation (-{100*scale_out_eff_hardware:.2f}%)"
                      )
    plt.text(x=0.5*(min_fom_node + min_fom_tile), 
             y = 1.03, 
             s=f"-{100*scale_up_gap:.2f}%s", 
             ha="center", rotation="vertical")



    plt.grid(zorder=0)
    plt.ylim([0,2.0])
    # plt.xlim([0.8*fom_scale_out, 1.2*numpy.max(fom_per_tile)])
    plt.legend(fancybox=True)
    plt.xlabel("Throughput [Img/s]")
    plt.gca().tick_params(labelleft=False)
    plt.title(title)
    plt.savefig(str(output_dir) + "/tile_FOM.pdf")

    return

def plot_scale_up_by_host(single_node_df, single_tile_df, gpus, title, output_dir):

    fig = plt.figure(figsize=(16,9))


    # Get the single-tile numbers:
    plt.scatter(single_tile_df['i_Host'], single_tile_df['Throughput'], 
        marker='o', color='black', label="Single-Tile")

    plt.xlabel("Node")
    plt.ylabel("Throughput [Img/s]")

    n_gpus = len(gpus)

    # Get the single-node numbers, normalized by n_gpus
    plt.scatter(single_node_df['i_Host'], single_node_df["Throughput"],
        marker="x", s=25, color='red', label="Full Node")

    plt.grid(True)
    plt.title(title)
    plt.legend()
    # plt.ylim([T_MIN, T_MAX])

    plt.savefig(str(output_dir) + "/scale_up_scatter.pdf")

    # Additionally, create a table of the efficiencies:
    # single_tile_df["Host-min"] = single_tile_df.groupby("Host")['Throughput'].transform(lambda x : x.min())
    # single_tile_df["Host-mean"] = single_tile_df.groupby("Host")['Throughput'].transform(lambda x : x.mean())

    mean_imgs = single_tile_df.groupby("Host")["Throughput"].mean()
    min_imgs  = single_tile_df.groupby("Host")["Throughput"].min()


    hosts = single_node_df["Host"]

    with open(output_dir / Path("eff.txt"), 'w') as f:
        f.write("host,min,min-prct,mean,mean-prct,scaleup,hardware-gap,comm-gap,hardware-prct,comm-prct\n")
        for i, h in enumerate(hosts):
            scale_up = single_node_df.iloc[i]['Throughput']
            min_p = scale_up / min_imgs[h]
            mean_p = scale_up / mean_imgs[h]
            output = f"{h},{min_imgs[h]},{min_p},{mean_imgs[h]},{mean_p},{scale_up}"

            # Also, write out the gap between hardware variation and communication
            # The mean imgs/s is what's achievable if all hardware is identically performing
            # The min img/s is the limiting hardware.
            # the gap between the two is how much is lost due to hardware variations
            # the rest of the gap for single-node scale up is due to communication overhead:
            total_gap = mean_imgs[h] - scale_up
            gap_hardware = mean_imgs[h] - min_imgs[h]
            gap_scale_up = min_imgs[h] - scale_up 

            output += f"{gap_hardware},{gap_scale_up},{gap_hardware/total_gap},{gap_scale_up/total_gap}"

            f.write(output + "\n")

    # plt.show()

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
    # plt.ylim([T_MIN, T_MAX])
    plt.grid(True)
    plt.title(title)
    plt.savefig(str(output_dir) + "/gpu_gpu_box.pdf")
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
    # plt.ylim([T_MIN, T_MAX])

    plt.savefig(str(output_dir) + "/gpu_gpu_scatter.pdf")
    # plt.show()

def plot_host_variation(df, title, output_dir):


    fig = plt.figure(figsize=(16,9))
    # ax = fig.add_subplot(projection='3d')

    plt.scatter(df['i_Host'], df['Throughput'], marker='o', color='black')

    plt.xlabel("Node")
    plt.ylabel("Throughput [Img/s]")
    plt.grid(True)
    plt.title(title)
    # plt.ylim([T_MIN, T_MAX])

    plt.savefig(str(output_dir) + "/host_variation.pdf")

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
    plt.savefig(str(output_dir) + "/tile_throughput.pdf")

    # plt.show()

def plot_tile_FOM(df, bins, title, output_dir, do_fit=False):


    norm = scipy.stats.norm

    fom_per_tile = df['Throughput'].values

    fig = plt.figure(figsize=(16,9))

    counts, bin_edges = numpy.histogram(fom_per_tile, bins=bins)
    min_fom = numpy.min(fom_per_tile)
    max_fom = numpy.max(fom_per_tile)

    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
    widths = bin_edges[1:] - bin_edges[:-1]

    # print(len(bins))
    # print(numpy.sum(bins))
    # print(widths)

    scale = widths[0]
    # print(scale)


    x = numpy.arange(numpy.min(bins),numpy.max(bins),0.1*widths[0])

    mean_fom = numpy.mean(fom_per_tile)


    plt.bar(bin_centers, counts, width=widths, zorder=3, 
        label=f"Min/max {min_fom:.2f}/{max_fom:.2f} ({100*(max_fom -min_fom)/max_fom:.2f}%)")
    
    if do_fit:
        fit = norm.fit(fom_per_tile)
        plt.plot(x, scale*norm(*fit).pdf(x), zorder=4, color="red", label=f"Gaussian ({fit[0]:.2f}, {fit[1]:.2f})")
    plt.grid(zorder=0)
    plt.legend()
    plt.xlabel("Throughput [Img/s]")
    plt.gca().tick_params(labelleft=False)
    plt.title(title)
    plt.savefig(str(output_dir) + "/tile_FOM.pdf")
    # plt.show()

if __name__ == "__main__":

    # This accepts the same arguments as the other script deliberately!

    parser = get_parser()

    args = parser.parse_args()
    # Convert the enums to the enum types:
    args.system = System[args.system.lower()]


    main(args)
