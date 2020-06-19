import larcv
import numpy
from matplotlib import pyplot as plt


def count_2d(file_name, product, producer):
    io = larcv.IOManager()
    io.add_in_file(file_name)
    io.initialize()
    n_classes = 3
    voxel_counts = {plane : numpy.zeros((io.get_n_entries(), n_classes)) for plane in [0,1,2] }

    energy = numpy.zeros((io.get_n_entries(),))
    flavor = numpy.zeros((io.get_n_entries(),))
    cc_nc  = numpy.zeros((io.get_n_entries(),))

    for i in range(io.get_n_entries()):
        io.read_entry(i)
        particles = io.get_data(product, producer)
        energy[i] = particles.at(0).energy_init()
        flavor[i] = particles.at(0).pdg_code()
        cc_nc[i]  = particles.at(0).nu_current_type()

        if i % 100 == 0:
            print("On entry ", i, " of ", io.get_n_entries())

        # if i > 1000:
        #     break

    # return energy, flavor, cc_nc
    return energy[0:i], flavor[0:i], cc_nc[0:i]


def plot(energy, flavor, cc_nc):

    bins = numpy.arange(0,3.21, 0.2)
    bin_centers = 0.5*(bins[1:] + bins[:-1])
    bin_widths  = bins[:-1] - bins[1:]

    cc_mask = cc_nc == 0


    total_events = len(energy)

    nue_mask  = numpy.bitwise_and(numpy.bitwise_or(flavor == 12 , flavor == -12), cc_mask)
    numu_mask = numpy.bitwise_and(numpy.bitwise_or(flavor == 14 , flavor == -14), cc_mask)
    nc_mask   = cc_nc == 1


    assert not(numpy.bitwise_and(nue_mask, numu_mask).any())
    assert not(numpy.bitwise_and(nc_mask, numu_mask).any())
    assert not(numpy.bitwise_and(nue_mask, nc_mask).any())

    nue_energy, _ = numpy.histogram(energy[nue_mask], bins=bins)
    numu_energy, _ = numpy.histogram(energy[numu_mask], bins=bins)
    nc_energy, _ = numpy.histogram(energy[nc_mask], bins=bins)

    nue_energy  = 100. * nue_energy / total_events # Converting to a percentage
    numu_energy = 100. * numu_energy / total_events # Converting to a percentage
    nc_energy   = 100. * nc_energy / total_events # Converting to a percentage

    print(nue_energy)
    print(numu_energy)
    print(nc_energy)

    plt.step(bin_centers-0.5*0.1, nue_energy, where='post', label=r"$\nu_e~$ C.C.")
    plt.step(bin_centers-0.5*0.1, numu_energy,where='post', label=r"$\nu_\mu~$ C.C.")
    plt.step(bin_centers-0.5*0.1, nc_energy,  where='post', label="N.C.")
    
    plt.grid(True)
    plt.xlabel("Neutrino Energy [GeV]")
    plt.ylabel("Fraction of Dataset [%]")
    plt.xlim([0,3.0])
    plt.legend()
    plt.show()

if __name__ == '__main__':
    energy, flavor, cc_nc = count_2d(
        "/Users/corey.adams/data/dlp_larcv3/sbnd_cosmic_samples/cosmic_tagging/cosmic_tagging_train.h5", 
        "particle", "sbndneutrino")

    plot(energy, flavor, cc_nc)

    # numpy.save("voxel_counts", vc)
    