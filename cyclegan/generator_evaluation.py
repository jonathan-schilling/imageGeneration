# lower score means more similarity between real and generated images
import pathlib
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import Model
from tensorflow.keras.layers import AveragePooling2D, Flatten
from tensorflow.python.keras.layers import Rescaling
import matplotlib.pyplot as plt
import numpy as np

import glob
import ntpath
import argparse
import os
import shutil

import pickle

from scipy.linalg import sqrtm

from skimage.transform import resize

from os import path

# Parameters #
from cyclegan.CycleGAN import define_generator

img_height = 144
img_width = 256
image_size = (img_height, img_width, 3)
z_size = 128

max_batches = 16

tf.get_logger().setLevel('ERROR')


def calculate_pd(model, image_input, image_output):
    act_output = model.predict(np.array([image_output])).numpy()
    act_input = model.predict(np.array([image_input])).numpy()

    difference = np.subtract(act_output, act_input)
    euclid = np.sum(np.square(difference))
    normalized = (1/np.prod(act_output.shape)) * euclid
    return normalized


def scale_images(images, new_shape):
    new_images = [resize(image, new_shape, 0) for image in images]  # TODO Test because of upscaling
    return np.asarray(new_images)


def get_dataset(path_, batch_size, image_size):
    data_dir = pathlib.Path(path_)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        seed=123,
        image_size=image_size[:2],
        batch_size=batch_size,
        crop_to_aspect_ratio=True,
        label_mode=None)

    normalization_layer = Rescaling(1. / 127.5, offset=-1)
    train_ds = train_ds.map(lambda x: normalization_layer(x))
    return train_ds


def generate_samples(generator_model, generator_path, number_of_samples, data, output_dim):
    generator_model.load_weights(generator_path)
    output = generator_model(data)
    scaled_data = scale_images(data, output_dim)
    scaled_output = scale_images(output, output_dim)
    return np.stack((scaled_data, scaled_output), axis=1)


def plot_fid_advc(epochs_used, epoch_fids, output):
    plt.clf()

    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(3 * len(epochs_used), 12))

    ax.boxplot(epoch_fids,
               vert=True,
               showmeans=True,
               meanline=True,
               labels=epochs_used)

    ax.yaxis.grid(True)
    ax.set_yscale('log')

    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Perception Distance', fontsize=14)

    plt.plot([], [], '--', linewidth=1, color='tab:green', label='mean')
    plt.plot([], [], '-', linewidth=1, color='tab:orange', label='median')
    plt.plot([], [], 'o', linewidth=1, color='k', label='outlier', fillstyle='none')
    plt.legend()

    plt.tight_layout()

    plt.savefig(path.join(output, 'plot_boxplot_fids.pdf'), dpi=300)
    plt.close()


def plot_fid(epochs_used, epoch_fids, output):
    plt.clf()

    plt.plot(epochs_used, np.median(epoch_fids, axis=1), label='median')
    plt.plot(epochs_used, np.mean(epoch_fids, axis=1), label='mean')

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Perception Distance', fontsize=12)
    plt.legend()

    plt.yscale('log')
    plt.xticks(epochs_used)
    plt.tight_layout()

    plt.savefig(path.join(output, 'plot_line_plot_fids.pdf'), dpi=300)
    plt.close()


def main(path_, generators_path, sample_size, output, generator_image_dim=(128, 128, 3), image_dim=(224, 224, 3)):
    vgg_16 = VGG16(weights='imagenet', include_top=True)
    vgg_16 = Model(inputs=vgg_16.input, outputs=[vgg_16.layers[15].output])
    data = next(iter(get_dataset(path_, sample_size, generator_image_dim)))
    epochs = []
    fids = []

    generator_model = define_generator()
    generator_model.build((1, *generator_image_dim))
    for file in os.listdir(generators_path):
        epochs.append(int(file.split("-")[1][:-3]))
        samples = generate_samples(generator_model, os.path.join(generators_path, file), sample_size, data, image_dim)
        current_fids = []
        for input_img, output_img in samples:
            current_fids.append(calculate_pd(vgg_16, input_img, output_img))
        fids.append(current_fids)
    plot_fid_advc(epochs, fids, output)
    plot_fid(epochs, fids, output)


if __name__ == '__main__':
    # Parse Arguments #
    parser = argparse.ArgumentParser(description='Evaluate CycleGAN')
    parser.add_argument('generators', type=str, help='Path where the gen models lie')
    parser.add_argument('samples', type=str, help='Path where the samples lie')
    parser.add_argument('-s', '--sampleSize', type=int, dest='sampleSize',
                        help='Sample Size of images that are used to calculate the FID. Default = 128', default=128)
    parser.add_argument('-o', '--output', type=str, dest="output", default="training",
                        help="The name of the image to (over-)write")

    args = parser.parse_args()
    main(args.samples, args.generators, args.sampleSize, args.output)
