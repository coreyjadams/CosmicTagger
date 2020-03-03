import torch
import numpy

class LossCalculator(torch.nn.Module):

    def __init__(self, balance_type=None):

        torch.nn.Module.__init__(self)


        if balance_type not in ["focal", "light", "even", "none"] and balance_type is not None:
            raise Exception("Unsupported loss balancing recieved: ", balance_type)

        self.balance_type = balance_type

        if balance_type != "none":
            self._criterion = torch.nn.CrossEntropyLoss(reduction='none')
        else:
            self._criterion = torch.nn.CrossEntropyLoss(reduction='mean')


    def label_counts(self, label_plane):
        # helper function to compute number of each type of label

        values, counts = torch.unique(label_plane, return_counts=True)


        # Make sure that if the number of counts is 0 for neutrinos, we fix that
        if len(counts) < 3:
            counts = torch.cat((counts, [1,]), 0 )

        return counts

    def forward(self, labels, logits):

        # This function receives the inputs labels and logits and returns a loss.\
        # If there is balancing scheme specified, weights are computed on the fly


        loss = None

        # labels and logits are by plane, loop over them:
        for i in [0,1,2]:
            plane_loss = self._criterion(input=logits[i], target=labels[i])
            if self.balance_type != "none":
                with torch.no_grad():
                    if self.balance_type == "focal":
                        # To compute the focal loss, we need to compute the one-hot labels and the
                        # softmax
                        softmax = torch.nn.functional.softmax(logits[i], dim=1)
                        # print("softmax.shape: ", softmax.shape)
                        # print("labels.shape: ", labels[i].shape)
                        onehot = torch.nn.functional.one_hot(labels[i], num_classes=3).float()
                        # print("onehot.shape: ", onehot.shape)
                        onehot = onehot.permute([0,3,1,2])
                        # print("onehot.shape: ", onehot.shape)

                        weights = onehot * (1 - softmax)**2
                        # print("weights.shape:  ", weights.shape)
                        weights = torch.mean(weights, dim=1)
                        # print("scale_factor.shape:  ", scale_factor.shape)
                        # print("plane_loss.shape: ", plane_loss.shape)
                        # scale_factor /= torch.mean(scale_factor)
                        # print("plane_loss.shape: ", plane_loss.shape)

                    elif self.balance_type == "even":
                        counts = self.label_counts(labels[i])
                        total_pixels = numpy.prod(labels[i].shape)
                        locs = torch.where(labels[i] != 0)
                        class_weights = 0.3333/(counts + 1.0)

                        weights = torch.full(labels[i].shape, class_weights[0])

                        weights[labels[i] == 1 ] = class_weights[1]
                        weights[labels[i] == 2 ] = class_weights[2]
                        pass

                    elif self.balance_type == "light":

                        total_pixels = numpy.prod(labels[i].shape)
                        per_pixel_weight = 1./(total_pixels)
                        weights = torch.full(labels[i].shape, per_pixel_weight)
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




