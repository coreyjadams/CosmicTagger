import torch


class AccuracyCalculator(object):

    def __init__(self):

        object.__init__(self)

    def segmentation_accuracy(self, labels, logits):


        accuracy = {}
        accuracy['Average/Total_Accuracy']   = 0.0
        accuracy['Average/Cosmic_IoU']       = 0.0
        accuracy['Average/Neutrino_IoU']     = 0.0
        accuracy['Average/Non_Bkg_Accuracy'] = 0.0
        accuracy['Average/mIoU']             = 0.0


        for plane in [0,1,2]:

            values, predicted_label = torch.max(logits[plane], dim=1)

            correct = (predicted_label == labels[plane].long()).float()

            # We calculate 4 metrics.
            # First is the mean accuracy over all pixels
            # Second is the intersection over union of all cosmic pixels
            # Third is the intersection over union of all neutrino pixels
            # Fourth is the accuracy of all non-zero pixels

            # This is more stable than the accuracy, since the union is unlikely to be ever 0

            non_zero_locations       = labels[plane] != 0

            weighted_accuracy = correct * non_zero_locations
            non_zero_accuracy = torch.sum(weighted_accuracy, dim=[1,2]) / torch.sum(non_zero_locations, dim=[1,2])

            neutrino_label_locations = labels[plane] == 2
            cosmic_label_locations   = labels[plane] == 1

            neutrino_prediction_locations = predicted_label == 2
            cosmic_prediction_locations   = predicted_label == 1


            neutrino_intersection = (neutrino_prediction_locations & \
                neutrino_label_locations).sum(dim=[1,2]).float()
            cosmic_intersection = (cosmic_prediction_locations & \
                cosmic_label_locations).sum(dim=[1,2]).float()

            neutrino_union        = (neutrino_prediction_locations | \
                neutrino_label_locations).sum(dim=[1,2]).float()
            cosmic_union        = (cosmic_prediction_locations | \
                cosmic_label_locations).sum(dim=[1,2]).float()
            # neutrino_intersection =

            one = torch.ones(1, dtype=neutrino_intersection.dtype,device=neutrino_intersection.device)


            neutrino_safe_unions = torch.where(neutrino_union != 0, True, False)
            neutrino_iou         = torch.where(neutrino_safe_unions, \
                neutrino_intersection / neutrino_union, one)

            cosmic_safe_unions = torch.where(cosmic_union != 0, True, False)
            cosmic_iou         = torch.where(cosmic_safe_unions, \
                cosmic_intersection / cosmic_union, one)

            # Finally, we do average over the batch

            cosmic_iou = torch.mean(cosmic_iou)
            neutrino_iou = torch.mean(neutrino_iou)
            non_zero_accuracy = torch.mean(non_zero_accuracy)

            accuracy[f'plane{plane}/Total_Accuracy']   = torch.mean(correct)
            accuracy[f'plane{plane}/Cosmic_IoU']       = cosmic_iou
            accuracy[f'plane{plane}/Neutrino_IoU']     = neutrino_iou
            accuracy[f'plane{plane}/Non_Bkg_Accuracy'] = non_zero_accuracy
            accuracy[f'plane{plane}/mIoU']             = 0.5*(cosmic_iou + neutrino_iou)

            accuracy['Average/Total_Accuracy']   += (0.3333333)*torch.mean(correct)
            accuracy['Average/Cosmic_IoU']       += (0.3333333)*cosmic_iou
            accuracy['Average/Neutrino_IoU']     += (0.3333333)*neutrino_iou
            accuracy['Average/Non_Bkg_Accuracy'] += (0.3333333)*non_zero_accuracy
            accuracy['Average/mIoU']             += (0.3333333)*(0.5)*(cosmic_iou + neutrino_iou)


        return accuracy


    def __call__(self, labels_dict, network_dict):

    	accuracy = self.segmentation_accuracy(labels_dict["segmentation"], network_dict["segmentation"])

    	return accuracy