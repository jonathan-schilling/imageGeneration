import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

import glob
import ntpath
import argparse

from os import path
from pathlib import Path

matplotlib.use('Agg')

# Parameters #
img_height = 144
img_width = 256
image_size = (img_height, img_width, 3)
z_size = 128

def create_samples(g_model, input_z, batch_size):
    g_output = g_model(input_z, training=False)
    images = tf.reshape(g_output, (batch_size, *image_size))
    return (images + 1) / 2.0

def output_results(batch_size, dir_path, every, output_image, start_epoch):

    model_path = path.join(dir_path, "models", "generator")

    mdls = glob.glob(path.join(model_path, "*.h5"))   
    mdls_existing = [int(ntpath.basename(y).split(".")[-2].replace("gen_model-","")) for y in mdls]
    mdls_existing.sort()
    mdls_used = [x for x in mdls_existing if x >= start_epoch]
    mdls_used = mdls_used[::every]
    
    epoch_samples = []
    
    fixed_z = tf.random.uniform(shape=(batch_size, z_size), minval=-1, maxval=1)

    for i, model in enumerate(mdls_used):
        print(f"\r Load Model {i}", end="", flush=True)
        gen_model = tf.keras.models.load_model(path.join(model_path, "gen_model-"+str(model)+".h5"))
        epoch_samples.append(create_samples(gen_model, fixed_z, batch_size).numpy())
        
    fig, axes = plt.subplots(figsize=(20, 5*len(epoch_samples)), nrows=len(epoch_samples), ncols=batch_size, sharex=True, sharey=True)
    for i,e in enumerate(epoch_samples):
        for j in range(batch_size):
            ax = axes[i,j]
            image = e[j]
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_title("Epoch:" + str(mdls_used[i]))
            plot_image(ax, image)
    
    fig.savefig(path.join(dir_path, output_image + ".pdf"))


de_normalization_layer = tf.keras.layers.Rescaling(1. / 2., offset=0.5)
def plot_image(ax, image):
    image = de_normalization_layer(image)
    ax.imshow(image)


if __name__ == '__main__':
    # Parse Arguments #
    parser = argparse.ArgumentParser(description='Train GAN to generate landscapes')
    parser.add_argument('every', type=int, help='Produce example for every xth checkpoint')
    parser.add_argument('-b', '--bSize', type=int, dest='bSize', help='Batch Size to use', default=3)
    parser.add_argument('-d', '--directory', type=str, dest="dirPath", default="training",
                        help="The output directory where the checkpoints and others are saved.")
    parser.add_argument('-o', '--output', type=str, dest="output", default="training",
                        help="The name of the image to (over-)write")
    parser.add_argument('-s', '--start', type=int, dest="start", default=0, help="Start at this epoch")

    args = parser.parse_args()
    output_results(args.bSize, args.dirPath, args.every, args.output, args.start)
