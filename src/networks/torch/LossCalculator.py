import torch
import numpy

from src.config import LossBalanceScheme

class LossCalculator(torch.nn.Module):

    def __init__(self, params):

        torch.nn.Module.__init__(self)


        # if balance_type not in ["focal", "light", "even", "none"] and balance_type is not None:
        #     raise Exception("Unsupported loss balancing recieved: ", balance_type)

        self.balance_type = params.loss_balance_scheme

        if self.balance_type != "none":
            self._criterion = torch.nn.CrossEntropyLoss(reduction='none')
        else:
            self._criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        self.weight_dict = {
            "segmentation" : params.seg_weight,
            "event_label"  : params.event_id_weight,
            "vertex"       : params.vertex_weight,
        }


        self.event_label_criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    def label_counts(self, label_plane):
        # helper function to compute number of each type of label

        values, counts = torch.unique(label_plane, return_counts=True)


        # Make sure that if the number of counts is 0 for neutrinos, we fix that
        if len(counts) < 3:
            counts = torch.cat((counts, [1,]), 0 )

        return counts

    def forward(self, labels_dict, network_dict):

        seg_loss   = self.segmentation_loss(labels_dict["segmentation"], network_dict["segmentation"])

        event_loss = self.event_loss(labels_dict["event_label"], network_dict["event_label"])
        
        vtx_loss   = self.vertex_loss(labels_dict["vertex"], network_dict["vertex"])

        return  \
            self.weight_dict["segmentation"] * seg_loss + \
            self.weight_dict["event_label"]  * event_loss + \
            self.weight_dict["vertex"]       * vtx_loss

    def vertex_loss(self, labels, logits):
        return 0.0

    def event_loss(self, labels, logits):

        event_label_loss = self.event_label_criterion(logits, labels.long())
        return event_label_loss

    def segmentation_loss(self, labels, logits):


        # This function receives the inputs labels and logits and returns a loss.\
        # If there is balancing scheme specified, weights are computed on the fly


        # Even if the model is half precision, we compute the loss and it's named_parameters
        # in full precision.

        loss = None

        # labels and logits are by plane, loop over them:
        for i in [0,1,2]:
            plane_loss = self._criterion(input=logits[i].float(), target=labels[i])
            if self.balance_type != LossBalanceScheme.none:
                with torch.no_grad():
                    if self.balance_type == LossBalanceScheme.focal:
                        # To compute the focal loss, we need to compute the one-hot labels and the
                        # softmax
                        softmax = torch.nn.functional.softmax(logits[i].float(), dim=1)
                        # print("torch.isnan(softmax).any(): ", torch.isnan(softmax).any())
                        onehot = torch.nn.functional.one_hot(labels[i], num_classes=3)
                        # print("torch.isnan(onehot).any(): ", torch.isnan(onehot).any())
                        onehot = onehot.permute([0,3,1,2])
                        # print("torch.isnan(onehot).any(): ", torch.isnan(onehot).any())
                        # print("onehot.shape: ", onehot.shape)

                        weights = onehot * (1 - softmax)**2
                        # print("torch.isnan(weights).any(): ", torch.isnan(weights).any())
                        # print("weights.shape:  ", weights.shape)
                        weights = torch.mean(weights, dim=1)
                        # print("torch.isnan(weights).any(): ", torch.isnan(weights).any())
                        # print("scale_factor.shape:  ", scale_factor.shape)
                        # print("plane_loss.shape: ", plane_loss.shape)
                        # scale_factor /= torch.mean(scale_factor)
                        # print("plane_loss.shape: ", plane_loss.shape)

                    elif self.balance_type == LossBalanceScheme.even:
                        counts = self.label_counts(labels[i])
                        total_pixels = numpy.prod(labels[i].shape)
                        locs = torch.where(labels[i] != 0)
                        class_weights = 0.3333/(counts + 1.0)

                        weights = torch.full(labels[i].shape, class_weights[0], device=labels[i].device)

                        weights[labels[i] == 1 ] = class_weights[1]
                        weights[labels[i] == 2 ] = class_weights[2]
                        pass

                    elif self.balance_type == LossBalanceScheme.light:

                        total_pixels = numpy.prod(labels[i].shape)
                        per_pixel_weight = 1.
                        weights = torch.full(labels[i].shape, per_pixel_weight, device=labels[i].device)
                        weights[labels[i] == 1 ] = 1.5 * per_pixel_weight
                        weights[labels[i] == 2 ] = 10  * per_pixel_weight

                total_weight = torch.sum(weights)
                plane_loss = torch.sum(weights*plane_loss)

                plane_loss /= total_weight

            if loss is None:
                loss = plane_loss
            else:
                loss += plane_loss


        return loss
