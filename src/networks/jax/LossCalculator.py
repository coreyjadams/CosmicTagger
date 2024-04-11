import jax
import jax.numpy as numpy

import optax

from src.config import LossBalanceScheme


def LossCalculator(params, weight = None):
    """ This is a closure pretending to be a class.
    It will return a function and not a real class ...
    """


    balance_type = params.mode.optimizer.loss_balance_scheme

    criterion = optax.softmax_cross_entropy_with_integer_labels

    event_label_criterion = optax.softmax_cross_entropy_with_integer_labels

    network_params = params.network



    def vertex_loss(labels, logits, event_label):


        target_dtype = logits[0].dtype
        # Weigh the loss:
        weight_detection    = numpy.asarray(network_params.vertex.l_det,
            dtype=target_dtype)
        weight_localization = numpy.asarray(network_params.vertex.l_coord,
            dtype=target_dtype)

        # This assumes channels last:
        detection_logits = [l[:,:,:,0] for l in logits]


        # Get the detection labels, and zero out any cosmic-only event
        detection_labels = [ l.astype(target_dtype) for l in labels['detection'] ]

        # # Compute pt, where CE=-log(pt)
        # pt = [ numpy.where(label == 1, logit, 1 - logit) for label, logit in zip(detection_labels, detection_logits) ]

        # # This is a computation of cross entropy
        # focus = [ weight_detection *(1 - _pt)**2 for _pt in pt]

        # bce_loss = [torch.nn.functional.binary_cross_entropy(logit, label, reduction="none")
        #     for logit, label in zip(detection_logits, detection_labels)
        # ]
        # focal_loss = [ f * l for f, l in zip(focus, bce_loss)]


        detection_loss = [
            optax.sigmoid_focal_loss(logits=_logit, labels=_label).sum(axis=(1,2))
            for _logit, _label in zip(detection_logits, detection_labels)
        ]
         


        # Compute the loss, don't sum over the batch index:
        # detection_loss = [ torch.sum(l,dim=(1,2)) for l in focal_loss]
        # This takes a mean over batches and over planes, scale it up so it's summing over planes:
        detection_loss = 3*numpy.mean(numpy.stack(detection_loss))
        # if focal_loss[0].device.index == 3:
        #     print("Detection Loss: ", detection_loss)


        regression_labels = numpy.split(labels['regression'], 3, axis=-1)
        # regression_labels = [torch.reshape(l, (-1, 2)) for l in regression_labels ]

        # Apply the sigmoid here:
        detection_localization_logits = [ jax.nn.sigmoid(l[:,:,:,1:]) for l in logits ]


        # This is (label - logit)**2 for every anchor box:
        # Since the label ranges from 0 to 1, and the logit ranges
        # from 0 to 1, this should map to 0 to 2.
        regression_loss = [
            (numpy.reshape(label, (-1,1,1,2))  - logits)**2
            for label, logits in zip(regression_labels, detection_localization_logits)
            ]

        has_vertex = event_label != 3

        # Sum over the channel axis to make it x^2 + y^2:
        regression_loss = [
            numpy.sum(l, axis=-1) for l in regression_loss
        ]
        # This step casts the label to be 0 or 1 in every anchor box,
        # but 0 in all boxes for events without a label.
        detection_labels = [ numpy.reshape(has_vertex, (-1,1,1))*d for d in detection_labels]

        # Scale each point by whether or not they have a label in them:
        regression_loss = [
            r_loss * d_label for r_loss, d_label in \
            zip(regression_loss, detection_labels)
        ]

        # Sum over all boxes:
        regression_loss = [
            numpy.sum(r_loss, axis=(1,2))
            for r_loss in regression_loss
        ]


        # Finally, take the sum over planes and mean over batches:
        regression_loss = numpy.stack(regression_loss, axis=-1)
        regression_loss = 3*weight_localization*numpy.sum(regression_loss)

        return detection_loss, regression_loss

    def event_loss_fn(labels, logits):
        if logits.dtype == numpy.float16:
           logits = logits.astype(numpy.float32)
        event_label_loss = event_label_criterion(logits, labels.astype(numpy.int32))
        return event_label_loss.mean()


    def segmentation_loss(labels, logits):


        # This function receives the inputs labels and logits and returns a loss.\
        # If there is balancing scheme specified, weights are computed on the fly


        # Even if the model is half precision, we compute the loss and it's named_parameters
        # in full precision.

        loss = None

        # labels and logits are by plane, loop over them:
        for i in [0,1,2]:
            plane_loss = criterion(logits=logits[i], labels=labels[i])
            # Block the gradients for weight calculations:
            logits[i] = jax.lax.stop_gradient(logits[i])
            if balance_type == LossBalanceScheme.focal:
                # To compute the focal loss, we need to compute the one-hot labels and the
                # softmax
                softmax = jax.nn.softmax(logits[i], axis=-1)
                # print("torch.isnan(softmax).any(): ", torch.isnan(softmax).any())
                onehot = jax.nn.one_hot(labels[i], num_classes=3)
                # print("torch.isnan(onehot).any(): ", torch.isnan(onehot).any())
                # onehot = onehot.permute([0,3,1,2])
                # print("torch.isnan(onehot).any(): ", torch.isnan(onehot).any())
                # print("onehot.shape: ", onehot.shape)

                weights = onehot * (1 - softmax)**2
                # print("torch.isnan(weights).any(): ", torch.isnan(weights).any())
                # print("weights.shape:  ", weights.shape)
                weights = numpy.mean(weights, axis=-1)
                # print("torch.isnan(weights).any(): ", torch.isnan(weights).any())
                # print("scale_factor.shape:  ", scale_factor.shape)
                # print("plane_loss.shape: ", plane_loss.shape)
                # scale_factor /= torch.mean(scale_factor)
                # print("plane_loss.shape: ", plane_loss.shape)

            elif balance_type == LossBalanceScheme.even:
                counts = label_counts(labels[i])
                total_pixels = numpy.prod(labels[i].shape)
                locs = torch.where(labels[i] != 0)
                class_weights = 0.3333/(counts + 1.0)

                weights = torch.full(labels[i].shape, class_weights[0], device=labels[i].device)

                weights[labels[i] == 1 ] = class_weights[1]
                weights[labels[i] == 2 ] = class_weights[2]
                pass

            elif balance_type == LossBalanceScheme.light:

                total_pixels = numpy.prod(labels[i].shape)
                per_pixel_weight = 1.
                weights = torch.full(labels[i].shape, per_pixel_weight, device=labels[i].device)
                weights[labels[i] == 1 ] = 1.5 * per_pixel_weight
                weights[labels[i] == 2 ] = 10  * per_pixel_weight
            else: # balance_type == LossBalanceScheme.none
                weights = 1.0

            plane_loss = numpy.mean(weights*plane_loss)

                # total_weight = torch.sum(weights)
                # plane_loss /= total_weight

            if loss is None:
                loss = plane_loss
            else:
                loss += plane_loss


        return loss

    def label_counts(label_plane):
        # helper function to compute number of each type of label

        values, counts = numpy.unique(label_plane, return_counts=True)


        # Make sure that if the number of counts is 0 for neutrinos, we fix that
        if len(counts) < 3:
            counts = numpy.concatenate((counts, [1,]), 0 )

        return counts
    

    def loss_fn(labels_dict, network_dict):


        # Cast to fp32 if currently in fp16:
        if network_dict["segmentation"][0].dtype == numpy.float16:
            network_dict["segmentation"] = \
                  [ n.astype(numpy.float32) for n in network_dict["segmentation"] ]
        loss = segmentation_loss(
            labels_dict["segmentation"],
            network_dict["segmentation"]
        )
        loss_metrics = {
            "loss/segmentation" : jax.lax.stop_gradient(loss)
        }

        if network_params.classification.active:
            event_loss = event_loss_fn(labels_dict["event_label"], network_dict["event_label"])
            event_loss = event_loss * network_params.classification.weight
            loss_metrics["event_label"] = jax.lax.stop_gradient(event_loss)
            loss =  loss +  event_loss

        if network_params.vertex.active:
            vtx_detection, vtx_localization = vertex_loss(
                labels_dict["vertex"], 
                network_dict["vertex"], 
                labels_dict['event_label'])
            vtx_detection  = network_params.vertex.weight * vtx_detection
            vtx_localization   = network_params.vertex.weight * vtx_localization
            loss_metrics["loss/vertex/detection"] = jax.lax.stop_gradient(vtx_detection)
            loss_metrics["loss/vertex/localization"] = jax.lax.stop_gradient(vtx_localization)

            loss = loss + vtx_detection + vtx_localization

        loss_metrics["loss/total"] = loss

        return  loss, loss_metrics

    return loss_fn

