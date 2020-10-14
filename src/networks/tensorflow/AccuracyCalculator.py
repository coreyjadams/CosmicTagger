import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils

import numpy

class AccuracyCalculator(object):

    def __init__(self):

        object.__init__(self)

    @tf.function
    def __call__(self, labels, prediction):

        # Labels is a list of tensors
        # Logits is a list of tensors
        with tf.compat.v1.variable_scope("accuracy"):
            n_planes = 3
            accuracies = {
                "total_accuracy"   : [None]*n_planes,
                "non_bkg_accuracy" : [None]*n_planes,
                "neut_iou"         : [None]*n_planes,
                "cosmic_iou"       : [None]*n_planes,
                "miou"             : [None]*n_planes
            }

            for p in range(n_planes):
            # for p in range(n_planes):

                # Accuracy over individual pixels:
                pixel_accuracy = tf.stop_gradient(tf.cast(tf.math.equal(labels[p], prediction[p]), dtype=tf.float32))

                accuracies["total_accuracy"][p] = tf.reduce_mean(pixel_accuracy)

                # Find the non zero labels:
                non_zero_indices = tf.cast(tf.not_equal(labels[p], tf.constant(0, labels[p].dtype)), tf.float32)



                weighted_accuracy = pixel_accuracy * non_zero_indices

                # Use non_zero_indexes to mask the accuracy to non zero label pixels
                accuracies["non_bkg_accuracy"][p] = tf.reduce_sum(weighted_accuracy) / tf.reduce_sum(non_zero_indices)


                # Neutrinos are label 2, cosmics label 1.  But iterate so I only need to
                # write these metrics once:

                for index in [1, 2]:
                    # Find the true indices:
                    label_indices       = tf.equal(
                        labels[p], tf.constant(index, labels[p].dtype))
                    # Find the predicted indices:
                    prediction_indices  = tf.equal(
                        prediction[p], tf.constant(index, prediction[p].dtype))


                    # To compute the intersection over union metrics,
                    # start with intersections and unions:
                    intersection = tf.math.logical_and(label_indices, prediction_indices)
                    union        = tf.math.logical_or(label_indices, prediction_indices)

                    iou = tf.reduce_sum(tf.cast(intersection, tf.float32)) / tf.reduce_sum(tf.cast(union, tf.float32))
                    if index == 1:
                        accuracies['cosmic_iou'][p] = iou
                    else :
                        accuracies['neut_iou'][p]   = iou

                accuracies['miou'][p] = 0.5*(accuracies['cosmic_iou'][p] + accuracies['neut_iou'][p])


            return accuracies
