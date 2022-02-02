import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils

import numpy

from src.config import LossBalanceScheme

class LossCalculator(object):

    def __init__(self, balance_type=None, channels_dim=1):

        object.__init__(self)


        # if balance_type not in ["focal", "light", "even", "none"] and balance_type is not None:
        #     raise Exception("Unsupported loss balancing recieved: ", balance_type)

        self.balance_type = balance_type
        self.channels_dim = channels_dim

        if balance_type != "none":
            self._criterion = tf.nn.sparse_softmax_cross_entropy_with_logits
        else:
            self._criterion = tf.nn.sparse_softmax_cross_entropy_with_logits


    def label_counts(self, label_plane):
        # helper function to compute number of each type of label

        counts = tf.math.bincount(tf.cast(label_plane, tf.int32), minlength=3,maxlength=3, dtype=tf.float32)

        return counts

    @tf.function
    def __call__(self, labels, logits):

        # This function receives the inputs labels and logits and returns a loss.\
        # If there is balancing scheme specified, weights are computed on the fly

        with tf.compat.v1.variable_scope('cross_entropy'):

            loss = None

            # labels and logits are by plane, loop over them:
            for i in [0,1,2]:
                plane_loss = self._criterion(labels=labels[i], logits=logits[i])
                if self.balance_type != LossBalanceScheme.none:
                    if self.balance_type == LossBalanceScheme.focal:

                        # Compute this as focal loss:
                        softmax = tf.nn.softmax(logits[i], axis = self.channels_dim)
                        one_hot = tf.one_hot(indices=labels[i], depth=3, axis=self.channels_dim)
                        weights = (1-softmax)**2
                        weights *= one_hot
                        weights = tf.reduce_sum(input_tensor=weights, axis=self.channels_dim)


                    elif self.balance_type == LossBalanceScheme.even:
                        counts = self.label_counts(labels[i])

                        class_weights = tf.constant(0.3333, dtype=tf.float32)/(counts + tf.constant(1.0, dtype=tf.float32))
                        weights = tf.fill(labels[i].shape, class_weights[0])
                        for i in [1,2]:
                            local_weights = tf.fill(labels[i].shape, class_weights[i])
                            weights = tf.where(labels[i] == i, local_weights, weights)

                        # weights[ ] = class_weights[1]
                        # weights[labels[i] == 2 ] = class_weights[2]
                        pass

                    elif self.balance_type == LossBalanceScheme.light:
                        total_pixels = numpy.prod(labels[i].get_shape().as_list())

                        weights = tf.fill(labels[i].shape, 1.)
                        for i in [1,2]:
                            if i == 1:
                                local_weights = tf.fill(labels[i].shape, 1.5)
                            else:
                                local_weights = tf.fill(labels[i].shape, 10.0)
                            weights = tf.where(labels[i] == i, local_weights, weights)


                    weights = tf.stop_gradient(weights)

                    # plane_loss = tf.reduce_mean(input_tensor=plane_loss)
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
