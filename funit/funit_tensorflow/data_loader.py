from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.layers import Rescaling
from tensorflow.python.data import AUTOTUNE
import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf


class Loader:

    def __init__(self, config) -> None:
        self.config = config
        self.content_dataset = self.create_dataset()
        self.content_dataset_iter = iter(self.content_dataset)
        self.class_dataset = self.create_dataset()
        self.class_dataset_iter = iter(self.class_dataset)
        self.test_dataset = self.create_testset()

    def create_dataset(self):
        dataset = image_dataset_from_directory(
            self.config['data_dir'],
            seed=123,
            image_size=self.config['img_size'][:2],
            batch_size=self.config['batch_size'],
            crop_to_aspect_ratio=True,
            )

        normalization_layer = Rescaling(1. / 127.5, offset=-1)
        dataset = dataset.map(lambda x, y: (normalization_layer(x), y))
        dataset = dataset.cache().shuffle(10000).prefetch(buffer_size=AUTOTUNE)
        return dataset

    def create_testset(self):
        dataset = image_dataset_from_directory(
            self.config['data_dir'],
            seed=123,
            batch_size=32,
            image_size=self.config['img_size'][:2],
            crop_to_aspect_ratio=True,
            )

        normalization_layer = Rescaling(1. / 127.5, offset=-1)
        dataset = dataset.map(lambda x, y: (normalization_layer(x), y))
        dataset = dataset.cache().shuffle(10000).prefetch(buffer_size=AUTOTUNE)
        return dataset

    def __iter__(self):
        self.content_dataset_iter = iter(self.content_dataset)
        self.class_dataset_iter = iter(self.class_dataset)
        return self

    def __next__(self):
        return next(self.content_dataset_iter), next(self.class_dataset_iter)

    def get_test_data(self, k=5):
        classes = {}
        single_image = None 
        for batch, label in self.test_dataset:
            for i, data in enumerate(batch):
                classes.setdefault(int(label[i]), list()).append(data)
            if all(map(lambda l: len(l) > k, classes.values())):
                for key, value in classes.items():
                    if single_image is None:
                        single_image = (value[0], key)
                    else:
                        return single_image, (np.array(value), key)