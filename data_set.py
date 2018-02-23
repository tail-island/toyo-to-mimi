import bz2
import numpy   as np
import os.path as path
import pickle

from funcy     import cat, mapcat, partition, repeat, repeatedly
from itertools import starmap
from utility   import child_paths


def load_data(data_path='./data'):
    def load_character(data_path):
        with bz2.open(path.join(data_path, 'x.pickle.bz2'), 'rb') as f:
            return pickle.load(f)

    def load_actor(data_path):
        return tuple(map(load_character, filter(path.isdir, child_paths(data_path))))

    def train_and_validate(waves):
        waves_train, waves_validate = zip(*map(lambda wave: (wave[22050 * 10:], wave[:22050 * 10]), waves))
        labels = range(len(waves))

        return (waves_train, labels), (np.array(tuple(mapcat(lambda wave: np.split(wave, 10), waves_validate))), np.array(tuple(mapcat(lambda label: repeat(label, 10), labels))))

    return train_and_validate(tuple(mapcat(load_actor, filter(path.isdir, child_paths(data_path)))))


def data_generator(waves, labels, batch_size):
    def to_samples(wave, label):
        start_index = int(np.random.random() * 22050)
        windows_size = (wave.shape[0] - start_index) // 22050

        return zip(np.split(wave[start_index:start_index + 22050 * windows_size], windows_size), repeat(label))

    def random_samples():
        return np.random.permutation(tuple(cat(starmap(to_samples, zip(waves, labels)))))

    def x_and_y(samples):
        return tuple(map(np.array, zip(*samples)))

    return map(x_and_y, partition(batch_size, cat(repeatedly(random_samples))))
