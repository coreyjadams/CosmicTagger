import numpy
from larcv import larcv

class AccuracyCalculator(object):

    def __init__(self, filename):
        object.__init__(self)


        self._io_manager = larcv.IOManager()
        self._io_manager.add_in_file(filename)
        self._io_manager.initialize()

        n_entries = self._io_manager.get_n_entries()

        self._total_accuracy = [
            numpy.zeros([n_entries]),
            numpy.zeros([n_entries]),
            numpy.zeros([n_entries]),
        ]

        self._active_accuracy = [
            numpy.zeros([n_entries]),
            numpy.zeros([n_entries]),
            numpy.zeros([n_entries]),
        ]

    def loop(self, max_i = 15):

        for i in range(self._io_manager.get_n_entries()):
            self._io_manager.read_entry(i)

            ev_image = larcv.EventSparseTensor2D.to_sparse_tensor(
                self._io_manager.get_data("sparse2d","sbndwire"))
            ev_label = larcv.EventSparseTensor2D.to_sparse_tensor(
                self._io_manager.get_data("sparse2d","sbnd_cosmicseg"))

            prediction_bkg  = larcv.EventImage2D.to_image2d(
                self._io_manager.get_data("image2d", "seg_bkg"))
            prediction_neut = larcv.EventImage2D.to_image2d(
                self._io_manager.get_data("image2d", "seg_neutrino"))
            prediction_cos  = larcv.EventImage2D.to_image2d(
                self._io_manager.get_data("image2d", "seg_cosmic"))

            print(prediction_bkg.as_vector().size())
            print(i)

            # Here, we have to compute some metrics.

            for plane in [0,1,2]:

                cos  = larcv.as_ndarray(prediction_cos.as_vector()[plane])
                neut = larcv.as_ndarray(prediction_neut.as_vector()[plane])
                bkg  = larcv.as_ndarray(prediction_bkg.as_vector()[plane])

                total_prediction = numpy.argmax(
                    numpy.stack(
                        [bkg, neut, cos]),
                    axis=0
                    )

                # Get the non zero labels:
                label_inds = ev_label.as_vector()[plane].indexes()
                label_vals = ev_label.as_vector()[plane].values()

                unraveled_label_inds = numpy.unravel_index(label_inds, shape=[640,1024])

                dense_label = numpy.zeros([640, 1024])
                dense_label[unraveled_label_inds] = label_vals

                print(f"Plane {plane} total accuracy:", numpy.mean(dense_label == total_prediction))
                self._total_accuracy[plane][i] = numpy.mean(dense_label == total_prediction)
                
                active_prediction = total_prediction[unraveled_label_inds]

                print(f"Plane {plane} active accuracy:", numpy.mean(active_prediction == label_vals))



            if i >= max_i:
                break


if __name__ == "__main__":

    acc_calc = AccuracyCalculator("/Users/corey.adams/data/dlp_larcv3/sbnd_cosmic_samples/cosmic_tagging_downsample_test_sparse_output.h5")
    acc_calc.loop()