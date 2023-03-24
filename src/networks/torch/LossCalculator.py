import torch
import numpy

from src.config import LossBalanceScheme

class LossCalculator(torch.nn.Module):

    def __init__(self, params, weight=None):

        torch.nn.Module.__init__(self)


        # if balance_type not in ["focal", "light", "even", "none"] and balance_type is not None:
        #     raise Exception("Unsupported loss balancing recieved: ", balance_type)

        self.balance_type = params.mode.optimizer.loss_balance_scheme

        self._criterion = torch.nn.CrossEntropyLoss(reduction='none')


        self.event_label_criterion = torch.nn.CrossEntropyLoss(reduction="mean", weight=weight)

        self.network_params = params.network


    def label_counts(self, label_plane):
        # helper function to compute number of each type of label

        values, counts = torch.unique(label_plane, return_counts=True)


        # Make sure that if the number of counts is 0 for neutrinos, we fix that
        if len(counts) < 3:
            counts = torch.cat((counts, [1,]), 0 )

        return counts

    def forward(self, labels_dict, network_dict):

        
        loss   = self.segmentation_loss(
            labels_dict["segmentation"],
            [ n.to(torch.float32) for n in network_dict["segmentation"] ]
        )
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

        target_dtype = logits[0].dtype
        # Weigh the loss:
        weight_detection    = torch.tensor(self.network_params.vertex.l_det,
            dtype=target_dtype, device=logits[0].device)
        weight_localization = torch.tensor(self.network_params.vertex.l_coord,
            dtype=target_dtype, device=logits[0].device)

        # This assumes channels first:
        detection_logits = [l[:,0,:,:] for l in logits]


        # Get the detection labels, and zero out any cosmic-only event
        detection_labels = labels['detection']
        # print("Min logit: ", [torch.min(l) for l in detection_logits])
        # print("Max logit: ", [torch.max(l) for l in detection_logits])

        # Compute pt, where CE=-log(pt)
        pt = [ torch.where(label == 1, logit, 1 - logit) for label, logit in zip(detection_labels, detection_logits) ]

        # This is a computation of cross entropy
        focus = [ weight_detection *(1 - _pt)**2 for _pt in pt]
        bce_loss = [torch.nn.functional.binary_cross_entropy(logit, label, reduction="none")
            for logit, label in zip(detection_logits, detection_labels)
        ]
        focal_loss = [ f * l for f, l in zip(focus, bce_loss)]
        # if focal_loss[0].device.index == 3:
            # print("label: ", detection_l?abels)
            # print("Min pt: ", [torch.min(l) for l in pt])
            # print("Max pt: ", [torch.max(l) for l in pt])
            # print("Min focal: ", [torch.min(l) for l in focal_loss])
            # print("Max focal: ", [torch.max(l) for l in focal_loss])
            # print(pt)

        has_vertex = event_label != 3

        # print("has_vertex: ",has_vertex)
        ## TODO:
        # convert this to binary cross entropy loss per-pixel
        # Then, it will work with focal loss better.


        detection_labels = [ torch.reshape(has_vertex, (-1,1,1))*d for d in detection_labels]

        # Compute the loss, don't sum over the batch index:
        detection_loss = [ torch.sum(l,dim=(1,2)) for l in focal_loss]
        # This takes a mean over batches and over planes, scale it up so it's summing over planes:
        detection_loss = 3*torch.mean(torch.stack(detection_loss))
        # if focal_loss[0].device.index == 3:
        #     print("Detection Loss: ", detection_loss)

        regression_labels = torch.chunk(labels['regression'], 3, dim=1)
        regression_labels = [torch.reshape(l, (-1, 2)) for l in regression_labels ]

        detection_localization_logits = [ l[:,1:,:,:] for l in logits ]

        # This is label - logit)**2 for every anchor box:
        # Since the label ranges from 0 to 1, and the logit ranges
        # from 0 to 1, this should map to 0 to 2.
        regression_loss = [
            (torch.reshape(label, (-1,2,1,1))  - logits)**2
            for label, logits in zip(regression_labels, detection_localization_logits)
            ]


        # Sum over the channel axis to make it x^2 + y^2:
        regression_loss = [
            torch.sum(l, axis=1) for l in regression_loss
        ]

        # Scale each point by whether or not they have a label in them:
        regression_loss = [
            r_loss * d_label for r_loss, d_label in \
            zip(regression_loss, detection_labels)
        ]

        # Sum over all boxes:
        regression_loss = [
            torch.sum(r_loss, axis=(1,2))
            for r_loss in regression_loss
        ]


        # Finally, take the sum over planes and mean over batches:
        regression_loss = torch.stack(regression_loss, axis=-1)
        regression_loss = 3*weight_localization*torch.sum(regression_loss)

        return detection_loss, regression_loss

    def event_loss(self, labels, logits):
        event_label_loss = self.event_label_criterion(logits.float(), labels.long())
        return event_label_loss

    def segmentation_loss(self, labels, logits):


        # This function receives the inputs labels and logits and returns a loss.\
        # If there is balancing scheme specified, weights are computed on the fly


        # Even if the model is half precision, we compute the loss and it's named_parameters
        # in full precision.

        loss = None

        # labels and logits are by plane, loop over them:
        for i in [0,1,2]:
            plane_loss = self._criterion(input=logits[i], target=labels[i])
            with torch.no_grad():
                if self.balance_type == LossBalanceScheme.focal:
                    # To compute the focal loss, we need to compute the one-hot labels and the
                    # softmax
                    softmax = torch.nn.functional.softmax(logits[i], dim=1)
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
                else: # self.balance_type == LossBalanceScheme.none
                    weights = 1.0

            plane_loss = torch.mean(weights*plane_loss)

                # total_weight = torch.sum(weights)
                # plane_loss /= total_weight

            if loss is None:
                loss = plane_loss
            else:
                loss += plane_loss


        return loss
