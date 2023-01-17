import numpy



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

class Decay:

    def __init__(self, start_value, floor, length, decay_rate):
        self.start_value = start_value
        self.floor       = floor
        self.length      = length
        self.decay_rate  = decay_rate


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        exp = numpy.exp(-self.decay_rate * (idx))
        return (self.start_value - self.floor) * exp + self.floor

    def __iter__(self):
        for i in range(self.length):
            yield self[i]


class LRSchedule:

    def __init__(self, segments):

        self.segments = segments


    def __len__(self):
        l = 0
        for segment in self.segments: l += len(segment)
        return l

    def __iter__(self):
        for segment in self.segments:
            for lr in segment:
                yield lr


    def __getitem__(self, idx):
        local_index = idx
        for segment in self.segments:
            if local_index >= len(segment):
                local_index -= len(segment)
            else:
                return segment[local_index]

        # Default return value:
        return 0.0
        
    def __call__(self, idx):
        return self.__getitem__(idx)

class WarmupFlatDecay(LRSchedule):

    def __init__(self, peak_learning_rate, decay_floor, epoch_length, decay_epochs, total_epochs):
        self.peak_learning_rate = peak_learning_rate
        self.decay_floor        = decay_floor
        self.epoch_length       = epoch_length
        self.decay_epochs       = decay_epochs
        if total_epochs is None:
            self.total_epochs = 25
        else:
            self.total_epochs       = total_epochs

        flat_epochs = self.total_epochs - self.decay_epochs - 1

        segments = [
            Linear(
                start_value = 0.0,
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

        super().__init__(segments)



class OneCycle(LRSchedule):

    def __init__(self, min_learning_rate, peak_learning_rate, decay_floor,
                       epoch_length, decay_epochs, total_epochs):
        self.min_learning_rate = min_learning_rate
        self.peak_learning_rate = peak_learning_rate
        self.decay_floor        = decay_floor
        self.epoch_length       = epoch_length
        self.decay_epochs       = decay_epochs
        self.total_epochs       = total_epochs

        triangle_epochs = total_epochs - decay_epochs
        total_steps = self.epoch_length * self.total_epochs
        decay_length = int(self.epoch_length * self.decay_epochs)
        up_length = int(0.5*triangle_epochs*self.epoch_length)
        down_length = total_steps - up_length - decay_length


        self.segments = [
            Linear(
                start_value = min_learning_rate,
                stop_value  = self.peak_learning_rate,
                length      = up_length
            ),
            Linear(
                start_value = self.peak_learning_rate,
                stop_value  = self.min_learning_rate,
                length      = down_length
            ),
            Decay(
                start_value = self.min_learning_rate,
                floor       = self.decay_floor,
                length      = decay_length,
                decay_rate  = 0.01,
            )
        ]


    def __len__(self):
        l = 0
        for segment in self.segments: l += len(segment)
        return l

    def __iter__(self):
        for segment in self.segments:
            for lr in segment:
                yield lr
