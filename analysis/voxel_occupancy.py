import larcv
import numpy



def count_2d(file_name, product, producer):
    io = larcv.IOManager()
    io.add_in_file(file_name)
    io.initialize()
    n_classes = 3
    voxel_counts = {plane : numpy.zeros((io.get_n_entries(), n_classes)) for plane in [0,1,2] }

    for i in range(io.get_n_entries()):
        io.read_entry(i)
        image = io.get_data(product, producer)
        for plane in [0,1,2]:
            sparse_tensor = image.at(plane)
            labels, counts = numpy.unique(sparse_tensor.values(), return_counts = True)
            for i_l, l in enumerate(labels):
                voxel_counts[plane][i][int(l)] = counts[i_l]
            # for label in range(n_classes):
            #     if label in labels:
            #         voxel_counts[plane][i][label] = counts[label]
            #     else:
            #         voxel_counts[plane][i][label] = 0

        if i % 100 == 0:
            print("On entry ", i, " of ", io.get_n_entries())

        # if i > 100:
        #     break

    print(voxel_counts)

    # print ("Average Voxel Occupation: ")
    # for p in [0,1,2]:
    #     print("  {p}: {av:.2f} +/- {rms:.2f} ({max} max)".format(
    #         p   = p, 
    #         av  = numpy.mean(voxel_counts[:,p]), 
    #         rms = numpy.std(voxel_counts[:,p]), 
    #         max = numpy.max(voxel_counts[:,p])
    #         )
    #     )

    return voxel_counts


if __name__ == '__main__':
    vc = count_2d("/Users/corey.adams/data/dlp_larcv3/sbnd_cosmic_samples/cosmic_tagging/cosmic_tagging_train.h5", "sparse2d", "sbnd_cosmicseg")

    numpy.save("voxel_counts", vc)
