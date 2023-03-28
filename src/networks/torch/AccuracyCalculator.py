import torch


class AccuracyCalculator(object):

    def __init__(self, params):

        object.__init__(self)

        self.network_params = params.network

    def segmentation_accuracy(self, labels, logits, batch_reduce=True):

        accuracy = {}
        accuracy['Average/Total_Accuracy']   = 0.0
        accuracy['Average/Cosmic_IoU']       = 0.0
        accuracy['Average/Neutrino_IoU']     = 0.0
        accuracy['Average/Non_Bkg_Accuracy'] = 0.0
        accuracy['Average/mIoU']             = 0.0

        # Compute accuracy in fp32!
        target_dtype  = torch.float32
        # target_dtype  = logits[0].dtype
        target_device = logits[0].device
        for plane in [0,1,2]:

            values, predicted_label = torch.max(logits[plane], dim=1)

            correct = (predicted_label == labels[plane].long()).type(target_dtype)
            # We calculate 4 metrics.
            # First is the mean accuracy over all pixels
            # Second is the intersection over union of all cosmic pixels
            # Third is the intersection over union of all neutrino pixels
            # Fourth is the accuracy of all non-zero pixels

            # This is more stable than the accuracy, since the union is unlikely to be ever 0

            non_zero_locations       = labels[plane] != 0

            # This must be done in fp32:
            weighted_accuracy = (correct * non_zero_locations).type(target_dtype)
            denom = torch.sum(non_zero_locations, dim=[1,2]).type(target_dtype) + 0.1
            non_zero_accuracy = torch.sum(weighted_accuracy, dim=[1,2]) / denom

            neutrino_label_locations = labels[plane] == 2
            cosmic_label_locations   = labels[plane] == 1

            neutrino_prediction_locations = predicted_label == 2
            cosmic_prediction_locations   = predicted_label == 1


            neutrino_intersection = (neutrino_prediction_locations & \
                neutrino_label_locations).sum(dim=[1,2]).type(target_dtype)
            cosmic_intersection = (cosmic_prediction_locations & \
                cosmic_label_locations).sum(dim=[1,2]).type(target_dtype)

            neutrino_union        = (neutrino_prediction_locations | \
                neutrino_label_locations).sum(dim=[1,2]).type(target_dtype)
            cosmic_union        = (cosmic_prediction_locations | \
                cosmic_label_locations).sum(dim=[1,2]).type(target_dtype)
            # neutrino_intersection =

            one = torch.ones(1, dtype=target_dtype,device=target_device)


            neutrino_safe_unions = torch.where(neutrino_union != 0, True, False)
            neutrino_iou         = torch.where(neutrino_safe_unions, \
                neutrino_intersection / neutrino_union, one)

            cosmic_safe_unions = torch.where(cosmic_union != 0, True, False)
            cosmic_iou         = torch.where(cosmic_safe_unions, \
                cosmic_intersection / cosmic_union, one)

            # Finally, we do average over the batch

            if batch_reduce:
                cosmic_iou = torch.mean(cosmic_iou)
                neutrino_iou = torch.mean(neutrino_iou)
                non_zero_accuracy = torch.mean(non_zero_accuracy)
                correct = torch.mean(correct)
            else:
                correct = torch.mean(correct, axis=(1,2))



            accuracy[f'plane{plane}/Total_Accuracy']   = correct
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

    def event_accuracy(self, label, logits, batch_reduce=True):

        selected_class = torch.argmax(logits, axis=-1)

        event_accuracy = selected_class == label

        event_accuracy = event_accuracy.type(logits[0].dtype)

        if batch_reduce:
            event_accuracy = torch.mean(event_accuracy)

        return {"Average/EventLabel" : event_accuracy}

    def vertex_accuracy(self, label, logits, predicted_vertex, event_label, batch_reduce=True):

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

    def __call__(self, network_dict, labels_dict, batch_reduce=True):

        with torch.no_grad():


            accuracy = self.segmentation_accuracy(labels_dict["segmentation"], network_dict["segmentation"], batch_reduce)



            if self.network_params.classification.active:
                accuracy.update(self.event_accuracy(labels_dict["event_label"], network_dict["event_label"], batch_reduce))



            if self.network_params.vertex.active:
                accuracy.update(self.vertex_accuracy(
                    labels_dict["vertex"], network_dict["vertex"],
                    network_dict["predicted_vertex"], labels_dict["event_label"], batch_reduce))



        return accuracy
