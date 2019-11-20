import numpy
from larcv import larcv
import time

# from matplotlib import pyplot as plt

COSMIC_LABEL_VALUE   = 2
NEUTRINO_LABEL_VALUE = 1

NUE_CC  = 0
NUMU_CC = 1
NC      = 2
COSMIC  = 3


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

        self.dtypes = numpy.dtype([
            ('entry'      , numpy.uint32),
            ('neut'       , numpy.uint32),
            ('n_neut_true', numpy.uint32,  3),
            ('n_neut_pred', numpy.uint32,  3),
            ('neut_x_mean', numpy.float32, 3),
            ('neut_y_mean', numpy.float32, 3),
            ('neut_x_std' , numpy.float32, 3),
            ('neut_y_std' , numpy.float32, 3),
            ('accuracy'   , numpy.float32, 3),
            ('acc_neut'   , numpy.float32, 3),
            ('acc_cosm'   , numpy.float32, 3),
            ('iou_neut'   , numpy.float32, 3),
            ('iou_cosm'   , numpy.float32, 3),
            ('acc_non0'   , numpy.float32, 3),
            ('energy'     , numpy.float32),
        ])

    def loop(self, max_i = 10):


        if max_i == -1:
            max_i = self._io_manager.get_n_entries()

        data = numpy.ndarray(max_i, dtype=self.dtypes)

        start = time.time()

        for i in range(self._io_manager.get_n_entries()):

            if i %25 == 0 and i != 0 :
                now = time.time()
                entries_per_second = i / (now - start + 0.1)
                print("Computing entry ", i, ", remaining estimated time: ", (max_i - i) / entries_per_second )

            if i >= max_i:
                break

            self._io_manager.read_entry(i)
            data[i]['entry'] = self._io_manager.current_entry()

            ev_image = larcv.EventSparseTensor2D.to_sparse_tensor(
                self._io_manager.get_data("sparse2d","sbndwire"))
            ev_label = larcv.EventSparseTensor2D.to_sparse_tensor(
                self._io_manager.get_data("sparse2d","sbnd_cosmicseg"))

            # prediction_bkg  = larcv.EventImage2D.to_image2d(
                # self._io_manager.get_data("image2d", "seg_bkg"))
            prediction_neut = larcv.EventSparseTensor2D.to_sparse_tensor(
                self._io_manager.get_data("sparse2d", "seg_neutrino"))
            prediction_cos  = larcv.EventSparseTensor2D.to_sparse_tensor(
                self._io_manager.get_data("sparse2d", "seg_cosmic"))

            # Get the neutrino particle type:
            if 'sbndneutrino' in self._io_manager.producer_list('particle'):
                neut = larcv.EventParticle.to_particle(self._io_manager.get_data("particle", "sbndneutrino"))
                neutrino = neut.as_vector().front()
                pdg    = neutrino.pdg_code()
                ccnc   = neutrino.nu_current_type()
                data[i]['energy'] = neutrino.energy_init()

                if ccnc == 1:
                    data[i]['neut'] = NC
                else:
                    if abs(pdg) == 12:
                        data[i]['neut'] = NUE_CC
                    else:
                        data[i]['neut'] = NUMU_CC
            else:
                data[i]['neut'] = COSMIC


            # Here, we have to compute some metrics.

            for plane in [0,1,2]:

                # cos  = larcv.as_ndarray(prediction_cos.as_vector()[plane])
                # neut = larcv.as_ndarray(prediction_neut.as_vector()[plane])
                # bkg  = larcv.as_ndarray(prediction_bkg.as_vector()[plane])

                pred_indexes = numpy.asarray(prediction_cos.as_vector()[plane].indexes())
                pred_values  = numpy.asarray(prediction_cos.as_vector()[plane].values())


                # Get the non zero labels:
                label_indexes = numpy.asarray(ev_label.as_vector()[plane].indexes())
                label_values  = numpy.asarray(ev_label.as_vector()[plane].values())

                # Compute the pixels that have neutrino/cosmic labels
                neutrino_truth = label_values == NEUTRINO_LABEL_VALUE
                cosmic_truth = label_values == COSMIC_LABEL_VALUE

                
                # print(numpy.sum(neutrino_truth))
                # print(numpy.sum(cosmic_truth))


                # This is the fraction of pixels that have the correct label, but only if the image pixel != 0
                correct     = label_values == pred_values

                # This is computing the location for each individual class:
                background_locs = label_values == 0
                cosmic_locs     = label_values == COSMIC_LABEL_VALUE
                neutrino_locs   = label_values == NEUTRINO_LABEL_VALUE

                # This computes the accuracy on just thos classes:
                background_acc = numpy.mean(correct[background_locs])
                cosmic_acc     = numpy.mean(correct[cosmic_locs])
                neutrino_acc   = numpy.mean(correct[neutrino_locs])

                # We also want to get the non background accuracy:
                non_background_acc = numpy.mean(correct[label_values != 0])

                # We want to get the IoU for neutrinos and cosmic pixels
                neutrino_prediction_locs = pred_values == NEUTRINO_LABEL_VALUE
                cosmic_prediction_locs   = pred_values == COSMIC_LABEL_VALUE


                neutrino_intersection = numpy.sum(numpy.logical_and(neutrino_prediction_locs, neutrino_truth))
                neutrino_union        = numpy.sum(numpy.logical_or(neutrino_prediction_locs, neutrino_truth))

                if neutrino_union < 1.0:
                    neutrino_iou = 0.5
                else:
                    neutrino_iou = neutrino_intersection  / neutrino_union


                cosmic_intersection = numpy.sum(numpy.logical_and(cosmic_prediction_locs, cosmic_truth))
                cosmic_union        = numpy.sum(numpy.logical_or(cosmic_prediction_locs, cosmic_truth))

                cosmic_iou = cosmic_intersection / cosmic_union
                
                # print()
                # print("Neutrino IoU: ", neutrino_iou)
                # print("Cosmic IoU: ", cosmic_iou)
                # print("Accuracy: ", numpy.mean(correct))
                # print("Background Accuracy: ", numpy.mean(correct[background_locs]))
                # print("Cosmic Accuracy: ", numpy.mean(correct[cosmic_locs]))
                # print("Neutrino Accuracy: ", numpy.mean(correct[neutrino_locs]))


                n_neutrino_pred = numpy.sum(neutrino_prediction_locs)
                neutrino_pred_x, neutrino_pred_y = numpy.unravel_index(pred_indexes[neutrino_prediction_locs], shape=[640,1024])
                neutrino_true_x, neutrino_true_y = numpy.unravel_index(label_indexes[neutrino_truth], shape=[640,1024])

                mean_x = numpy.mean(neutrino_true_x)
                mean_y = numpy.mean(neutrino_true_y)

                std_x = numpy.std(neutrino_true_x)
                std_y = numpy.std(neutrino_true_y)                


                data[i]['n_neut_true'][plane] = numpy.sum(neutrino_locs)
                data[i]['n_neut_pred'][plane] = numpy.sum(neutrino_prediction_locs)
                data[i]['neut_x_mean'][plane] = mean_x
                data[i]['neut_y_mean'][plane] = mean_y
                data[i]['neut_x_std'][plane]  = std_x
                data[i]['neut_y_std'][plane]  = std_y
                data[i]['accuracy'][plane]    = numpy.mean(correct)
                data[i]['acc_neut'][plane]    = numpy.mean(correct[neutrino_locs])
                data[i]['acc_cosm'][plane]    = numpy.mean(correct[cosmic_locs])
                data[i]['iou_neut'][plane]    = neutrino_iou
                data[i]['iou_cosm'][plane]    = cosmic_iou
                data[i]['acc_non0'][plane]    = numpy.mean(correct[label_values != 0])

        return data

if __name__ == "__main__":
    folder = "/Users/corey.adams//data/cosmic_tagging_downsample/"
    # _file  = "cosmic_tagging_downsample_test_sparse_output_biggerbatch.h5"
    # _file  = "cosmic_tagging_downsample_test_sparse_output_biggerbatch_2.h5"
    _file  = "cosmic_tagging_downsample_test_sparse_output_baseline_fullbalance_2.h5"
    # _file  = "cosmic_tagging_downsample_dev_sparse_output.h5"
    acc_calc = AccuracyCalculator(folder + _file)
    d = acc_calc.loop(max_i = 7000)
    # print(d)

    numpy.save(folder + _file.replace(".h5", ".npy"), d)

