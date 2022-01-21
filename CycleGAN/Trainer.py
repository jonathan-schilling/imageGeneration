import tensorflow as tf
import numpy as np
import os
import pathlib
from WGAN import WGAN

img_height, img_width = 72, 128


def get_dataset(data, batch_size):
    data_dir = pathlib.Path(data)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        crop_to_aspect_ratio=True)

    normalization_layer = tf.keras.layers.Rescaling(1. / 127.5, offset=-1)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    train_ds = train_ds.cache().shuffle(1500)
    return train_ds


wgan = WGAN(get_dataset("bilderNeuro", 32), (img_height, img_width, 3), 32, 5)

wgan.train(10)
