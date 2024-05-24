import tensorflow as tf
from tensorflow import keras

""" 
This file isn't my proudest work but I'm really just trying to staple this together.

TF changed their interface and I'm just trying to make it work!
"""

class Linear():

    def __init__(self, start_value, stop_value, length):

        self.start = start_value
        self.stop  = stop_value
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        for i in range(self.length):
            yield self[i]

    def __getitem__(self, idx):
        return self.start +  idx *(self.stop - self.start) / self.length

    def __repr__(self):
        return f"Linear from {self.start} to {self.stop} for length {self.length}.\n"

class Flat():

    def __init__(self, start_value, length):

        self.start = start_value
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        for i in range(self.length):
            yield self.start

    def __getitem__(self, idx):
        return self.start

    def __repr__(self):
        return f"Flat at {self.start} for length {self.length}.\n"


class Decay:

    def __init__(self, start_value, floor, length, decay_rate):
        self.start       = start_value
        self.floor       = floor
        self.length      = length
        self.decay_rate  = decay_rate


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        exp = numpy.exp(-self.decay_rate * (idx))
        return (self.start - self.floor) * exp + self.floor

    def __iter__(self):
        for i in range(self.length):
            yield self[i]

    def __repr__(self):
        return f"Decay from {self.start} to {self.floor} for length {self.length} at rate {self.decay_rate}.\n"



class FlatSchedule(keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, start_value, epoch_length, total_epochs):
        if total_epochs is None:
            total_epochs = 25

        self.start_value = start_value
        self.epoch_length = epoch_length
        self.total_epochs = total_epochs


    def __call__(self, idx):
        return self.start_value
    
    def __len__(self):
        return self.epoch_length * self.total_epochs



class WarmupFlatDecay(keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, peak_learning_rate, decay_floor, epoch_length, decay_epochs, total_epochs):
        self.peak_learning_rate = peak_learning_rate
        self.decay_floor        = decay_floor
        self.epoch_length       = epoch_length
        self.decay_epochs       = decay_epochs
        if total_epochs is None:
            self.total_epochs = 25
        else:
            self.total_epochs       = total_epochs

        self.flat_epochs = self.total_epochs - self.decay_epochs - 1

        segments = [
            Linear(
                start_value = 1e-6,
                stop_value  = self.peak_learning_rate,
                length      = self.epoch_length
            ),
            Flat(
                start_value = self.peak_learning_rate,
                length      = flat_epochs * self.epoch_length,
            ),
            Decay(
                start_value = self.peak_learning_rate,
                floor       = self.decay_floor,
                length      = self.epoch_length * self.decay_epochs,
                decay_rate  = 0.01,
            )
        ]


    def __call__(self, idx):
        # Conditionals to determine where we are:
        if idx < self.epoch_length:
            return 1e-6 +  idx *(self.peak_learning_rate - 1e-6) / self.epoch_length
        if idx <  self.flat_epochs * self.epoch_length:
            return self.peak_learning_rate
        else:
            exp = tf.exp(- 0.01 * (idx))
            return (self.peak_learning_rate - self.decay_floor) * exp + self.decay_floor

# class OneCycle(LRSchedule):

#     def __init__(self, min_learning_rate, peak_learning_rate, decay_floor,
#                        epoch_length, decay_epochs, total_epochs):
#         self.min_learning_rate = min_learning_rate
#         self.peak_learning_rate = peak_learning_rate
#         self.decay_floor        = decay_floor
#         self.epoch_length       = epoch_length
#         self.decay_epochs       = decay_epochs
#         self.total_epochs       = total_epochs

#         triangle_epochs = total_epochs - decay_epochs
#         total_steps = self.epoch_length * self.total_epochs
#         decay_length = int(self.epoch_length * self.decay_epochs)
#         up_length = int(0.5*triangle_epochs*self.epoch_length)
#         down_length = total_steps - up_length - decay_length


#         self.segments = [
#             Linear(
#                 start_value = min_learning_rate,
#                 stop_value  = self.peak_learning_rate,
#                 length      = up_length
#             ),
#             Linear(
#                 start_value = self.peak_learning_rate,
#                 stop_value  = self.min_learning_rate,
#                 length      = down_length
#             ),
#             Decay(
#                 start_value = self.min_learning_rate,
#                 floor       = self.decay_floor,
#                 length      = decay_length,
#                 decay_rate  = 0.01,
#             )
#         ]


#     def __len__(self):
#         l = 0
#         for segment in self.segments: l += len(segment)
#         return l

#     def __iter__(self):
#         for segment in self.segments:
#             for lr in segment:
#                 yield lr
