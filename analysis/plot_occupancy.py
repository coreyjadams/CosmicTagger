import numpy
from matplotlib import pyplot as plt

def load_data():

    return numpy.load("voxel_counts.npy", allow_pickle=True).item()


if __name__ == "__main__":
    counts = load_data()


    min_bin = -4.5
    max_bin = 0.5
    n_bins = 50
    bins = numpy.logspace(min_bin, max_bin, n_bins)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    bin_widths = bins[1:] - bins[:-1]
    total_pixels = 2048 * 1280*3 / 100.
    n_events = 43075 * 0.01
    # for plane in [0,1,2]:

    fig = plt.figure(figsize=(9,6))


    hists = []
    for label in [0,1,2]:
        label_counts = counts[0][:,label] + counts[1][:,label] + counts[2][:,label]
        label_counts /= total_pixels 
        label_histogram, bin_edge = numpy.histogram(label_counts, bins)
        label_histogram = label_histogram.astype(numpy.float) / n_events
        hists.append(label_histogram)


    # plt.plot(bin_centers, hists[0], label="background")
    plt.bar(bin_centers, hists[1], width=bin_widths, label="Cosmic Pixels"  , alpha=0.5 )
    plt.bar(bin_centers, hists[2], width=bin_widths, label="Neutrino Pixels", alpha=0.5 )


    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(15) 
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(15) 

    plt.grid(True)
    plt.legend(fontsize=20)
    plt.xlabel("Percentage of pixels / event", fontsize=15)
    plt.ylabel("Fraction of Dataset [%]", fontsize=15)
    plt.xlim([10**-4, 10**1])
    plt.gca().set_xscale("log")
    plt.tight_layout()
    plt.show()
