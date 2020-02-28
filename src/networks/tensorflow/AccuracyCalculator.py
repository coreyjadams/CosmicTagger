import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils

import numpy

class AccuracyCalculator(object):

    def __init__(self):

        object.__init__(self)


    def __call__(self, labels, prediction):

        # Labels is a list of tensors
        # Logits is a list of tensors
        with tf.name_scope("Accuracy"):
            n_planes = 3
            accuracies = {
                "total_accuracy"   : [None]*n_planes,
                "non_bkg_accuracy" : [None]*n_planes,
                "neut_iou"         : [None]*n_planes,
                "cosmic_iou"       : [None]*n_planes
            }

            for p in range(n_planes):

                # Accuracy over individual pixels:
                pixel_accuracy = tf.stop_gradient(tf.cast(tf.math.equal(labels[p], prediction[p]), dtype=tf.float16))

                accuracies["total_accuracy"][p] = tf.reduce_mean(pixel_accuracy)

                # Find the non zero labels:
                non_zero_indices = tf.not_equal(labels[p], tf.constant(0, labels[p].dtype))


                # Use non_zero_indexes to mask the accuracy to non zero label pixels
                accuracies["non_bkg_accuracy"][p] = tf.reduce_mean(tf.boolean_mask(pixel_accuracy, non_zero_indices))


                # Neutrinos are label 2, cosmics label 1.  But iterate so I only need to
                # write these metrics once:

                for index in [1, 2]:
                    # Find the true indices:
                    label_indices       = labels[p] == index

                    # Find the predicted indices:
                    prediction_indices  = prediction[p] == index


                    # To compute the intersection over union metrics,
                    # start with intersections and unions:

                    intersection = tf.math.logical_and(label_indices, prediction_indices)
                    union        = tf.math.logical_or(label_indices, prediction_indices)

                    iou = tf.reduce_sum(tf.cast(intersection, tf.float32)) / tf.reduce_sum(tf.cast(union, tf.float32))
                    if index == 1:
                        accuracies['cosmic_iou'][p] = iou
                    else :
                        accuracies['neut_iou'][p]   = iou



            return accuracies
