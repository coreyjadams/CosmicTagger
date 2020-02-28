import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils

import numpy

class LossCalculator(object):

    def __init__(self, balance_type=None, channels_dim=1):

        object.__init__(self)


        if balance_type not in ["focal", "light", "even", "none"] and balance_type is not None:
            raise Exception("Unsupported loss balancing recieved: ", balance_type)

        self.balance_type = balance_type
        self.channels_dim = channels_dim

        if balance_type != "none":
            self._criterion = tf.nn.sparse_softmax_cross_entropy_with_logits
        else:
            self._criterion = tf.nn.sparse_softmax_cross_entropy_with_logits


    def label_counts(self, label_plane):
        # helper function to compute number of each type of label

        y, idx, counts = tf.unique_with_counts(label_plane)

        print(counts)
        # Make sure that if the number of counts is 0 for neutrinos, we fix that
        if len(counts.shape) < 3:
            counts = tf.concat((counts, [1,]), 0 )

        return counts

    def __call__(self, labels, logits):

        # This function receives the inputs labels and logits and returns a loss.\
        # If there is balancing scheme specified, weights are computed on the fly

        with tf.compat.v1.name_scope('cross_entropy'):

            loss = None

            # labels and logits are by plane, loop over them:
            for i in [0,1,2]:
                plane_loss = self._criterion(labels=labels[i], logits=logits[i])
                if self.balance_type != "none":
                    if self.balance_type == "focal":

                        # Compute this as focal loss:
                        softmax = tf.nn.softmax(logits[i], axis = self.channels_dim)
                        ont_hot = tf.one_hot(indices=labels[i], depth=3, axis=self.channels_dim)

                        weights = (1-softmax)**2
                        weights *= ont_hot
                        weights = tf.reduce_sum(input_tensor=weights, axis=self.channels_dim)


                    elif self.balance_type == "even":
                        counts = self.label_counts(labels[i])
                        total_pixels = numpy.prod(labels[i].shape)
                        locs = tf.compat.v1.where(labels[i] != 0)
                        class_weights = 0.3333/(counts + 1)

                        weights = tf.full(labels[i].shape, class_weights[0])

                        weights[labels[i] == 1 ] = class_weights[1]
                        weights[labels[i] == 2 ] = class_weights[2]
                        pass

                    elif self.balance_type == "light":
                        print(labels[i].shape)
                        print(labels[i].get_shape())
                        total_pixels = numpy.prod(labels[i].get_shape())
                        print(total_pixels)
                        print(type(total_pixels))   
                        per_pixel_weight = 1./(total_pixels)
                        print(per_pixel_weight)
                        weights = tf.full(labels[i].shape, per_pixel_weight, dytpe=tf.float32)
                        weights[labels[i] == 1 ] = 1.5 * per_pixel_weight
                        weights[labels[i] == 2 ] = 10  * per_pixel_weight

                    weights = tf.stop_gradient(weights)

                    plane_loss *= weights
                    plane_loss = tf.reduce_mean(input_tensor=plane_loss)
                    total_weight = tf.reduce_sum(weights)


                    plane_loss = tf.reduce_sum(weights*plane_loss)

                    plane_loss /= total_weight
                else:
                    plane_loss = tf.reduce_mean(plane_loss)
                if loss is None:
                    loss = plane_loss
                else:
                    loss += plane_loss

            return loss
