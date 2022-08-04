import pathlib
import matplotlib
from matplotlib import pyplot as plt
import numpy

from datetime import datetime

font = {'family' : 'Times',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)

def parse_log_file(fname):

    # All logs have 500 lines

    times = []
    iteration = []
    io    = []
    step  = []

    fom = None
    previous_time = None
    with open(fname) as _f:
        i = 0
        lines = _f.readlines()
        for line in lines:
            if "Total time to batch process except first two iterations" in line:
                fom = float(line.split(" ")[-1])
            if "train Step" not in line: continue
            # split the line into timings
            tokens = line.split(" ")
            # First token is the date
            # Second token is the timestamp
            t = datetime.strptime(tokens[1], "%H:%M:%S,%f")
            if previous_time is None:
                times.append( (t - t).total_seconds())
            else:
                times.append( (t - previous_time).total_seconds())

            previous_time = t
            # times.append(tokens[1])
            # -2 token is step time
            # -5 token is IO time
            # -8 token is instantaneous Img/s
            step.append(float(tokens[-2]))
            io.append(float(tokens[-5].replace('(', '')))
            iteration.append(int(tokens[7]))
#                 {"Time"   : pd.to_datetime(tokens[1]),
#                  "Step"   : tokens[7],
#                  "Img/s"  : tokens[-8],
#                  "IO"     : tokens[-5],
#                  "Step"   : tokens[-2],
#                 },


            i += 1
            if i > 500: break

    times = numpy.asarray(times, dtype=numpy.float32)
    # times = numpy.asarray(times[1:] - times[0:-1], dtype=numpy.float32 ) / (1000.*1000.)# convert to millseconds
    # print(times[1:] - times[0:-1])

    io = numpy.asarray(io)[2:]
    step = numpy.asarray(step)[2:]

    # Skip the first two steps, which is just the first duration
    return numpy.asarray(iteration[:-2]), times[2:], io, step, fom

def analyze_iteration(arr):
    print(f"  Mean time: {numpy.mean(arr):.3f} +/- {numpy.std(arr):.3f}")
    print(f"  Median time: {numpy.median(arr):.3f}")
    print(f"  Max time: {numpy.max(arr):.3f} ({numpy.argmax(arr)})")
    print(f"  50/75/90/95 quantile: {numpy.quantile(arr, [0.5, 0.75, 0.9, 0.95])}")


def plot_data():
    pass


def fetch_log_from_reframe(SIZE, JOBID):

    SIZE=f"{SIZE:03d}"
    reframe_top = pathlib.Path("/lus/grand/projects/PolarisAT/reframe_output/output/cosmictagger/")
    subpath = "{JOBID}/polaris/compute/builtin/COSMICTAGGER_{SIZE}/rfm_COSMICTAGGER_{SIZE}_job.out"

    full_path = reframe_top / pathlib.Path(SIZE) / pathlib.Path(subpath.format(JOBID=JOBID, SIZE=SIZE))

    return full_path

def plot_run_data(plot_name, title, iteration, _times, _io, _step_time):
    # How much time in forward pass?
    residual = _times

    print("IO: ")
    analyze_iteration(_io)
    print("Step: ")
    analyze_iteration(_step_time)
    print("Forward: (estimated)")
    analyze_iteration(residual)
    print("Durations: ")
    analyze_iteration(_times)


    # Create histograms for the various timings.
    fig = plt.figure(figsize=(16,9))

    biggest_m = None

    for array, label, color in zip(
        [residual, _io, _step_time],
        ["Total", "IO", "Backwards"],
        ["green", "black", "red"]
        ):


        # This is the biggest time for this run.
        m = numpy.min([numpy.max([numpy.max(array), 1.0]), 2.3])
        # Bins for the histogram:
        bins = numpy.arange(0,1.01*m, 0.01)

        if biggest_m is None:
            biggest_m = m
        else:
            if m > biggest_m:
                biggest_m = m

        hist, bin_edges = numpy.histogram(array, bins=bins)
        # Make it a normalized percentage:
        hist = 100.*hist / len(array)

        # Compute the middle locations:
        bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
        bin_widths = bin_edges[1:] - bin_edges[:-1]

        # Plot the timing info:
        plt.plot(bin_centers, hist, lw=2, label = label, color = color)


        # Compute quantiles:
        quant_bins = numpy.arange(0,1.01, 0.01)
        # print(quant_bins)
        quantiles = numpy.quantile(array, quant_bins)
        # print(quantiles)

        # Add one more bin to the quantiles to make it finish the plot:
        quant_bins = numpy.append(quant_bins, 1.00)
        quantiles  = numpy.append(quantiles, m+0.1)


        plt.plot(quantiles, 100*quant_bins, lw=1, ls="--", color = color)


    plt.xlabel("Time [s]", fontsize=36)
    plt.ylabel("Normalized Distribution [%]", fontsize=36)
    plt.grid(True)
    plt.ylim([0,101])
    plt.xlim([0, 2.5])
    plt.legend(fontsize=36)
    plt.title(title, fontsize=30)
    plt.savefig(f"{plot_name}.pdf")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--log-dir', type=pathlib.Path,
                        help='Directory Location of the profiles to analyze')
    parser.add_argument('--sum', dest='accumulate', action='store_const',
                        const=sum, default=max,
                        help='sum the integers (default: find the max)')

    args = parser.parse_args()
    main(args)


# # ThetaGPU results
# log_top = pathlib.Path("/home/cadams/ThetaGPU/CosmicTagger/output/torch/polaris/")
# folder = pathlib.Path("thetagpu_polaris_baremetal")
# filename = pathlib.Path("process.log")
#
# Things that ran on Milan-PolarisAT
log_top = pathlib.Path("/lus/grand/projects/PolarisAT/COSMICTAGGER/CosmicTagger/output/torch/polaris/")
folder = pathlib.Path("at-baremetal-2022-06-08_512nodes-affinity-depth-open-network/")
filename = pathlib.Path("process.log")
s=512

# Rahki results
# log_top = pathlib.Path("/home/cadams/CT_Acceptance_analysis")
# folder = pathlib.Path("logs_bare_metal")
# filename = pathlib.Path("512nodes-train-conda-torch.qsub.o242534")

full_path = log_top / folder / filename
print(full_path)
#
# s = 512
# j = 24
#
# full_path = fetch_log_from_reframe(SIZE=s, JOBID=j)



iteration, times, io, step_time, fom = parse_log_file(full_path)

title=f"Polaris Milan AT: {4*s} GPUs, Container, FOM: {fom:.1f}"
plot_run_data(f"polaris-milanAT-conda-open-nccl", title, iteration[10:], times[10:], io[10:], step_time[10:])
print("FOM: ", fom)
