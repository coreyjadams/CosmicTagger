import jax
from jax import numpy



def event_accuracy(label, logits, batch_reduce=True):

    selected_class = numpy.argmax(logits, axis=-1)

    event_accuracy = selected_class == label

    event_accuracy = event_accuracy.astype(logits[0].dtype)

    if batch_reduce:
        event_accuracy = numpy.mean(event_accuracy)

    return {"Average/EventLabel" : event_accuracy}


def segmentation_accuracy(labels, logits, batch_reduce=True):

    accuracy = {}
    accuracy['Average/Total_Accuracy']   = 0.0
    accuracy['Average/Cosmic_IoU']       = 0.0
    accuracy['Average/Neutrino_IoU']     = 0.0
    accuracy['Average/Non_Bkg_Accuracy'] = 0.0
    accuracy['Average/mIoU']             = 0.0

    # Compute accuracy in fp32!
    target_dtype  = numpy.float32
    # target_dtype  = logits[0].dtype

    for plane in [0,1,2]:
        predicted_label = numpy.argmax(logits[plane], axis=-1)

        correct = (predicted_label == labels[plane].astype(target_dtype))

        # We calculate 4 metrics.
        # First is the mean accuracy over all pixels
        # Second is the intersection over union of all cosmic pixels
        # Third is the intersection over union of all neutrino pixels
        # Fourth is the accuracy of all non-zero pixels

        # This is more stable than the accuracy, since the union is unlikely to be ever 0

        non_zero_locations       = labels[plane] != 0

        # This must be done in fp32:
        weighted_accuracy = (correct * non_zero_locations)

        # Sum but not over the batch dim:
        denom = numpy.sum(non_zero_locations, axis=[1,2]) + 0.1

        non_zero_accuracy = numpy.sum(weighted_accuracy, axis=[1,2]) / denom

        neutrino_label_locations = labels[plane] == 2
        cosmic_label_locations   = labels[plane] == 1

        neutrino_prediction_locations = predicted_label == 2
        cosmic_prediction_locations   = predicted_label == 1


        neutrino_intersection = (neutrino_prediction_locations & \
            neutrino_label_locations).sum(axis=[1,2]).astype(target_dtype)
        cosmic_intersection = (cosmic_prediction_locations & \
            cosmic_label_locations).sum(axis=[1,2]).astype(target_dtype)

        neutrino_union        = (neutrino_prediction_locations | \
            neutrino_label_locations).sum(axis=[1,2]).astype(target_dtype)
        cosmic_union        = (cosmic_prediction_locations | \
            cosmic_label_locations).sum(axis=[1,2]).astype(target_dtype)
        # neutrino_intersection =



        neutrino_safe_unions = numpy.where(neutrino_union != 0, True, False)
        neutrino_iou         = numpy.where(neutrino_safe_unions, \
            neutrino_intersection / neutrino_union, 1.0)

        cosmic_safe_unions = numpy.where(cosmic_union != 0, True, False)
        cosmic_iou         = numpy.where(cosmic_safe_unions, \
            cosmic_intersection / cosmic_union, 1.0)

        # Finally, we do average over the batch

        if batch_reduce:
            cosmic_iou = numpy.mean(cosmic_iou)
            neutrino_iou = numpy.mean(neutrino_iou)
            non_zero_accuracy = numpy.mean(non_zero_accuracy)
            correct = numpy.mean(correct)
        else:
            correct = numpy.mean(correct, axis=(1,2))



        accuracy[f'plane{plane}/Total_Accuracy']   = correct
        accuracy[f'plane{plane}/Cosmic_IoU']       = cosmic_iou
        accuracy[f'plane{plane}/Neutrino_IoU']     = neutrino_iou
        accuracy[f'plane{plane}/Non_Bkg_Accuracy'] = non_zero_accuracy
        accuracy[f'plane{plane}/mIoU']             = 0.5*(cosmic_iou + neutrino_iou)

        accuracy['Average/Total_Accuracy']   += (0.3333333)*numpy.mean(correct)
        accuracy['Average/Cosmic_IoU']       += (0.3333333)*cosmic_iou
        accuracy['Average/Neutrino_IoU']     += (0.3333333)*neutrino_iou
        accuracy['Average/Non_Bkg_Accuracy'] += (0.3333333)*non_zero_accuracy
        accuracy['Average/mIoU']             += (0.3333333)*(0.5)*(cosmic_iou + neutrino_iou)

    return accuracy


def AccuracyCalculator(params):

    network_params = params.network



    def accuracy_fn(network_dict, labels_dict, batch_reduce=True):



        accuracy = segmentation_accuracy(
            labels_dict["segmentation"], 
            network_dict["segmentation"], 
            batch_reduce
        )


        if network_params.classification.active:
            accuracy.update(
                event_accuracy(
                    labels_dict["event_label"], 
                    network_dict["event_label"], 
                    batch_reduce
                )
            )



        if network_params.vertex.active:
            accuracy.update(
                vertex_accuracy(
                    labels_dict["vertex"], 
                    network_dict["vertex"],
                    network_dict["predicted_vertex"], 
                    labels_dict["event_label"], 
                    batch_reduce
                )
            )

        return accuracy

    return accuracy_fn





def vertex_accuracy(label, logits, predicted_vertex, event_label, batch_reduce=True):

    accuracy = {}
    target_dtype = logits[0].dtype

    detection_logits = [l[:,0,:,:] for l in logits]

    accuracy['Average/VertexDetection']  = 0.0
    accuracy['Average/VertexResolution']  = 0.0

    #flatten to make argmax easier:

    batch_size = logits[0].shape[0]


    detection_logits = [d.reshape(batch_size, -1) for d in detection_logits]
    detection_labels = [d.reshape(batch_size, -1) for d in label['detection']]

    true_index  = [torch.argmax(d.type(target_dtype), dim=1) for d in detection_labels ]
    predicted_index = [torch.argmax(d.type(target_dtype), dim=1) for d in detection_logits ]

    equal = [s == p for s, p in zip(true_index, predicted_index)]

    detection_accuracy = [ e.type(target_dtype) for e in equal ]
    if batch_reduce:
        detection_accuracy = [ torch.mean(e) for e in detection_accuracy ]

    # print("Predicted: ", predicted_vertex)
    # print("Actual: ", label['xy_loc'])


    difference = (label['xy_loc'] - predicted_vertex)**2

    difference = torch.sqrt(torch.sum(difference, axis=-1))

    has_vertex = (event_label != 3).type(target_dtype)


    difference = difference * has_vertex.reshape((-1,1))
    difference = difference.type(target_dtype)

    # Average over batch:
    if batch_reduce:
        difference = torch.sum(difference, axis=0) / ( torch.sum(has_vertex) + 1e-5)


    if batch_reduce:
        for p, d in enumerate(detection_accuracy):
            accuracy[f"plane{p}/VertexDetection"] = d
            accuracy[f"plane{p}/VertexResolution"] = difference[p]
            accuracy["Average/VertexDetection"] += 0.3333333*d
            accuracy["Average/VertexResolution"] += 0.3333333*difference[p]

    else:
        for p, d in enumerate(detection_accuracy):
            accuracy[f"plane{p}/VertexDetection"] = d
            accuracy[f"plane{p}/VertexResolution"] = difference[:,p]
            accuracy["Average/VertexDetection"] += 0.3333333*d
            accuracy["Average/VertexResolution"] += 0.3333333*difference[:,p]


    return accuracy
