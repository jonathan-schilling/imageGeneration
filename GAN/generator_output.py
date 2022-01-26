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
import matplotlib.plt as plt
import argparse
import ntpath

# Parameters #

img_height = 72
img_width = 128
image_size = (72, 128, 3)
z_size = 128

def create_samples(g_model, input_z, batch_size):
    g_output = g_model(input_z, training=False)
    images = tf.reshape(g_output, (batch_size, *image_size))
    return (images + 1) / 2.0

def output_results(batch_size, checkpoints, epochs, every, output_image, start_epoch):
    if tf.test.is_gpu_available():
        device_name = '/GPU:0'
    else:
        device_name = '/CPU:0'

    print(device_name)

    gen_checkpoint_path = checkpoints + "/generator/" + "/{epoch:04d}.ckpt"
    gen_checkpoint_dir = os.path.dirname(gen_checkpoint_path)
    gen_model = tf.keras.models.load_model(gen_checkpoint_dir + "/gen-model")
    gen_model.summary()

    epoch_samples = []

    fixed_z = tf.random.uniform(shape=(batch_size, z_size), minval=-1, maxval=1)

    chps = glob.glob(gen_checkpoint_dir + "/*index")
    batches_existing = [int(ntpath.basename(y).split(".")[-3]) for y in chps]
    batches_used = []

    n = 0
    for i,checkpoint in enumerate(chps):
        if i % every == 0 and batches_existing[n] >= start_epoch:
            gen_model.load_weights(gen_checkpoint_dir + "/" + Path(checkpoint).stem)
            epoch_samples.append(create_samples(gen_model, fixed_z, batch_size).numpy())
            batches_used.append(batches_existing[n])
        n += 1

    fig, axes = plt.subplots(figsize=(20, 5*len(epoch_samples)), nrows=len(epoch_samples), ncols=batch_size, sharex=True, sharey=True)
    for i,e in enumerate(epoch_samples):
        for j in range(batch_size):
            ax = axes[i,j]
            image = e[j]
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_title(batches_used[i])
            plot_image(ax, image)
    
    fig.savefig(output_image + ".pdf")

de_normalization_layer = tf.keras.layers.Rescaling(1. / 2., offset=0.5)
def plot_image(ax, image):
    image = de_normalization_layer(image)
    ax.imshow(image)


if __name__ == '__main__':
    # Parse Arguments #
    parser = argparse.ArgumentParser(description='Train GAN to generate landscapes')
    parser.add_argument('every', type=int, help='Produce example for every xth checkpoint')
    parser.add_argument('epochs', type=int, help='Epochs available')
    parser.add_argument('-b', '--bSize', type=int, dest='bSize', help='Batch Size to use', default=3)
    parser.add_argument('-c', '--checkpoints', type=str, dest="checkpoints", default="training",
                        help="The output directory where the checkpoints are saved.")
    parser.add_argument('-o', '--output', type=str, dest="output", default="training",
                        help="The name of the image to (over-)write")
    parser.add_argument('-s', '--start', type=int, dest="start", default=0, help="Start at this epoch")

    args = parser.parse_args()
    output_results(args.bSize, args.checkpoints, args.epochs, args.every, args.output, args.start)
