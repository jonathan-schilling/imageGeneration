import argparse

import tensorflow as tf
import numpy as np
import os
import pathlib
from WGAN import WGAN

img_height, img_width = 144, 256


def get_dataset(data, batch_size):
    data_dir = pathlib.Path(data)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        seed=123,
        labels=None,
        label_mode=None,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        follow_links=True,
        crop_to_aspect_ratio=True)

    normalization_layer = tf.keras.layers.Rescaling(1. / 127.5, offset=-1)
    train_ds = train_ds.map(lambda x: (normalization_layer(x)))
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    return train_ds


if __name__ == '__main__':
    # Parse Arguments #
    parser = argparse.ArgumentParser(description='Train Wasserstein GAN to generate landscapes')
    parser.add_argument('bSize', type=int, help='Batch Size to use')
    parser.add_argument('epochs', type=int, help='Number of epochs to train')
    parser.add_argument('-d', '--directory', type=str, dest="path", default="training",
                        help="The output directory where the checkpoints are saved. It will be created if it dosen't "
                             "exist and overritten (!) if it does.")
    parser.add_argument('-c', '--checkpoints', type=int, dest="chps", default=5,
                        help='Take checkpoint every x epochs. Default = 5')
    parser.add_argument('-ct', '--continue', dest='continue_', action='store_true', default=False,
                        help="Continue training (default: Start from the beginning)")

    args = parser.parse_args()

    wgan = WGAN(get_dataset("bilderNeuro", args.bSize), (img_height, img_width, 3), args.bSize, 5, path_like=args.path,
                load=args.continue_, save_interval=args.chps)
    wgan.train(args.epochs)