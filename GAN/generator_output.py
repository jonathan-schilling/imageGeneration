import tensorflow as tf
import numpy as np
import os
import PIL
import PIL.Image
import pathlib
import matplotlib
import glob
from pathlib import Path

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

# Parameters #

img_height = 72
img_width = 128
image_size = (72, 128, 3)
z_size = 128


def output_results(batch_size, checkpoints, epochs, every, output_image):
    if tf.test.is_gpu_available():
        device_name = '/GPU:0'
    else:
        device_name = '/CPU:0'

    print(device_name)

    gen_model = tf.keras.models.load_model(checkpoints + "/gen-model")
    gen_model.summary()
    gen_checkpoint_path = checkpoints + "/gen-{epoch:04d}.ckpt"
    gen_checkpoint_dir = os.path.dirname(gen_checkpoint_path)

    def create_samples(g_model, input_z):
        g_output = g_model(input_z, training=False)
        images = tf.reshape(g_output, (batch_size, *image_size))
        return (images + 1) / 2.0

    epoch_samples = []

    fixed_z = tf.random.uniform(shape=(batch_size, z_size), minval=-1, maxval=1)

    chps = glob.glob(gen_checkpoint_dir + "/gen-*index")

    for i,checkpoint in enumerate(chps):
        if i % every == 0:
            gen_model.load_weights(gen_checkpoint_dir + "/" + Path(checkpoint).stem)
            epoch_samples.append(create_samples(gen_model, fixed_z).numpy())

    de_normalization_layer = tf.keras.layers.Rescaling(1. / 2., offset=0.5)

    fig = plt.figure(figsize=(10, 14))
    n = 0
    for i in range(len(chps)):
        if i % every == 0:
            for j in range(3):
                ax = fig.add_subplot(epochs // every, 3, n * 3 + j + 1)
                image = epoch_samples[n][j]
                image = de_normalization_layer(image)
                ax.imshow(image)
            n += 1
    fig.savefig(output_image)


if __name__ == '__main__':
    # Parse Arguments #
    parser = argparse.ArgumentParser(description='Train GAN to generate landscapes')
    parser.add_argument('every', type=int, help='Produce example for every xth checkpoint')
    parser.add_argument('bSize', type=int, help='Batch Size to use')
    parser.add_argument('epochs', type=int, help='Epochs available')
    parser.add_argument('-c', '--checkpoints', type=str, dest="checkpoints", default="training",
                        help="The output directory where the checkpoints are saved.")
    parser.add_argument('-o', '--output', type=str, dest="output", default="training",
                        help="The name of the image to (over-)write")

    args = parser.parse_args()
    output_results(args.bSize, args.checkpoints, args.epochs, args.every, args.output)
