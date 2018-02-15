import os

from collections           import ChainMap
from funcy                 import butlast, concat, partial, repeatedly
from keras                 import backend as K
from keras.engine.topology import Layer
from operator              import attrgetter


def child_paths(directory):
    return sorted(map(attrgetter('path'), os.scandir(directory)))


class ZeroPadding(Layer):
    def __init__(self, output_channel_size, **kwargs):
        self.output_channel_size = output_channel_size
        super(ZeroPadding, self).__init__(**kwargs)

    def build(self, input_shape):
        assert self.output_channel_size % input_shape[-1] == 0

        self.input_channel_size = input_shape[-1]
        super(ZeroPadding, self).build(input_shape)

    def call(self, x):
        if self.input_channel_size == self.output_channel_size:
            return x
        else:
            return K.concatenate((x, K.concatenate(tuple(repeatedly(partial(K.zeros_like, x), self.output_channel_size // self.input_channel_size - 1)))))

    def compute_output_shape(self, input_shape):
        return tuple(concat(butlast(input_shape), (self.output_channel_size,)))

    def get_config(self):
        return dict(ChainMap(super(ZeroPadding, self).get_config(), {'output_channel_size': self.output_channel_size}))
