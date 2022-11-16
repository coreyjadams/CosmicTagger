import torch
import numpy

from src.config import LossBalanceScheme

class LossCalculator(torch.nn.Module):

    def __init__(self, params):

        torch.nn.Module.__init__(self)


        # if balance_type not in ["focal", "light", "even", "none"] and balance_type is not None:
        #     raise Exception("Unsupported loss balancing recieved: ", balance_type)

        self.balance_type = params.mode.optimizer.loss_balance_scheme

        if self.balance_type != "none":
            self._criterion = torch.nn.CrossEntropyLoss(reduction='none')
        else:
            self._criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        self.event_label_criterion = torch.nn.CrossEntropyLoss(reduction="mean")

        self.network_params = params.network


    def label_counts(self, label_plane):
        # helper function to compute number of each type of label

        values, counts = torch.unique(label_plane, return_counts=True)


        # Make sure that if the number of counts is 0 for neutrinos, we fix that
        if len(counts) < 3:
            counts = torch.cat((counts, [1,]), 0 )

        return counts

    def forward(self, labels_dict, network_dict):

        loss   = self.segmentation_loss(labels_dict["segmentation"], network_dict["segmentation"])
        loss_metrics = {
            "segmentation" : loss.detach()
        }

        if self.network_params.classification.active:
            event_loss = self.event_loss(labels_dict["event_label"], network_dict["event_label"])
            event_loss = event_loss * self.network_params.classification.weight
            loss_metrics["event_label"] = event_loss.detach()
            loss =  loss +  event_loss

        if self.network_params.vertex.active:
            vtx_detection, vtx_localization = self.vertex_loss(labels_dict["vertex"], network_dict["vertex"], labels_dict['event_label'])
            vtx_detection  = self.network_params.vertex.weight * vtx_detection
            vtx_localization   = self.network_params.vertex.weight * vtx_localization
            loss_metrics["vertex/detection"] = vtx_detection.detach()
            loss_metrics["vertex/localization"] = vtx_localization.detach()

            loss      +=  vtx_detection + vtx_localization

        loss_metrics["total"] = loss

        return  loss, loss_metrics

    def vertex_loss(self, labels, logits, event_label):

        # This assumes channels first:
        detection_logits = [l[:,0,:,:] for l in logits]

        # Get the detection labels, and zero out any cosmic-only event
        detection_labels = labels['detection']
        has_vertex = event_label != 3


        detection_labels = [ torch.reshape(has_vertex, (-1,1,1))*d for d in detection_labels]


        # The YOLO detection loss is scaled by the no-obj parameter.
        # Compute the loss per anchor box:
        detection_loss = [
            torch.nn.functional.mse_loss(i.float(), t.float(), reduction="none")
            # torch.nn.functional.cross_entropy(i.float(), t.float(), reduction="mean")
            for i, t in zip(detection_logits, labels['detection'])
        ]

        # Compute the weight per object box:
        # Assuming the lambda_noobj parameter is small compared to 1, this is approximately correct
        weights = [l + self.network_params.vertex.l_noobj for l in labels['detection']]

        # Compute the weight * loss:
        detection_loss = [ torch.mean(l * w) for l, w in zip(detection_loss, weights)]
        detection_loss = torch.sum(torch.stack(detection_loss))


        # # For localization, we first compute the index locations of active vertexes:
        # active_sites = [ d == 1 for d in detection_labels]

        # detection_localization_logits = [ l[:,1:,:,:] for l in logits ]

        # regression_labels = torch.chunk(labels['regression'], 3, dim=1)
        # regression_labels = [torch.reshape(l, (-1, 2)) for l in regression_labels ]

        # labels_x = [rl[:,0][has_vertex] for rl in regression_labels]
        # labels_y = [rl[:,1][has_vertex] for rl in regression_labels]




        # # Split x and y, and then zero in on the active sites:
        # detection_localization_logits_x = [ l[:,1,:,:] for l in logits ]
        # detection_localization_logits_y = [ l[:,2,:,:] for l in logits ]

        # # Select the active sites in labels and logits:
        # active_localization_logits_x = [ d[a] for d, a in zip(detection_localization_logits_x, active_sites) ]
        # active_localization_logits_y = [ d[a] for d, a in zip(detection_localization_logits_y, active_sites) ]

        # # Weigh the loss:
        # weight = self.network_params.vertex.l_coord

        # # Compute the loss in x and y:
        # loss_x = [ weight*torch.nn.functional.mse_loss(lx, alx) for lx, alx in zip(labels_x, active_localization_logits_x)]
        # loss_y = [ weight*torch.nn.functional.mse_loss(ly, aly) for ly, aly in zip(labels_y, active_localization_logits_y)]

        # # Sum the losses:
        # localization_loss = [x + y for x, y in zip(loss_x, loss_y)]
        # localization_loss = torch.sum(torch.stack(localization_loss))

        return detection_loss, detection_loss
        # return detection_loss, localization_loss

    def event_loss(self, labels, logits):
        # print(logits.shape)
        # print(labels.shape)
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

                plane_loss = torch.mean(weights*plane_loss)

                # total_weight = torch.sum(weights)
                # plane_loss /= total_weight

            if loss is None:
                loss = plane_loss
            else:
                loss += plane_loss


        return loss
