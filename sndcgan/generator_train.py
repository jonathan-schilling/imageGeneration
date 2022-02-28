import tensorflow as tf
from tensorflow.train import Checkpoint, CheckpointManager
import numpy as np
import os
from os import path
import pickle
from time import time, strftime, gmtime
import pathlib
import shutil

from tensorflow.python.data import AUTOTUNE

import csv
from generator_output import plot_image, create_samples

import matplotlib.pyplot as plt
import argparse
import time
import ntpath

tf.random.set_seed(1)
np.random.seed(1)

# Parameters #

img_height = 144
img_width = 256
image_size = (img_height, img_width, 3)
z_size = 128


# Generator #

def make_dcgan_generator(output_size):
    n_filters = 512
    hidden_size = (output_size[0] // 8, output_size[1] // 8)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128,)),

        tf.keras.layers.Dense(units=512 * np.prod(hidden_size), use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Reshape((hidden_size[0], hidden_size[1], 512)),

        tf.keras.layers.Conv2DTranspose(
            filters=256, kernel_size=(4, 4),
            strides=(2, 2), padding='same', use_bias=False
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2DTranspose(
            filters=128, kernel_size=(4, 4),
            strides=(2, 2), padding='same', use_bias=False
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2DTranspose(
            filters=64, kernel_size=(4, 4),
            strides=(2, 2), padding='same', use_bias=False
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2DTranspose(
            filters=3, kernel_size=(3, 3),
            strides=(1, 1), padding='same', use_bias=False,
            activation='tanh'
        )
    ])

    return model


# Discriminator #

def make_dcgan_discriminator(dropout_rate, input_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_size),

        tf.keras.layers.Conv2D(
            filters=64, kernel_size=(3, 3),
            strides=(1, 1), padding='same'
        ),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.Conv2D(
            filters=128, kernel_size=(4, 4),
            strides=(2, 2), padding='same'
        ),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3),
            strides=(1, 1), padding='same'
        ),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.Conv2D(
            filters=256, kernel_size=(4, 4),
            strides=(2, 2), padding='same'
        ),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.Conv2D(
            filters=256, kernel_size=(3, 3),
            strides=(1, 1), padding='same'
        ),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.Conv2D(
            filters=512, kernel_size=(4, 4),
            strides=(2, 2), padding='same'
        ),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.Conv2D(
            filters=512, kernel_size=(3, 3),
            strides=(1, 1), padding='same'
        ),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
    ])

    return model


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
    train_ds = train_ds.cache().shuffle(10000).prefetch(AUTOTUNE)
    return train_ds


def display_examples(samples, number_of_images, output_image, info_text):
    figure = plt.figure(figsize=(20, 10))
    for j in range(number_of_images):
        ax = figure.add_subplot(1, number_of_images, j + 1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        image = samples[j]
        plot_image(ax, image)
    figure.suptitle(info_text, size="xx-large")
    figure.savefig(output_image + ".pdf")
    plt.close(figure)


# create a line plot of loss for the gan and save to file
def plot_history(dir_path, losses):
    # plot history
    plt.clf()
    for key, val in losses.items():
        plt.plot(val, label=key)
    plt.legend()
    plt.savefig(path.join(dir_path, 'plot_line_plot_loss.png'))
    plt.close()

#@tf.function  # speed up code
def train_step(input_real, input_z, gen_model, disc_model, loss_fn, g_optimizer, d_optimizer):
    with tf.GradientTape() as g_tape:
        g_output = gen_model(input_z)
        d_logits_fake = disc_model(g_output, training=True)
        labels_real = tf.ones_like(d_logits_fake)
        g_loss = loss_fn(y_true=labels_real, y_pred=d_logits_fake)

        g_grads = g_tape.gradient(g_loss, gen_model.trainable_variables)
        g_optimizer.apply_gradients(grads_and_vars=zip(g_grads, gen_model.trainable_variables))

    with tf.GradientTape() as d_tape1:
        d_logits_real = disc_model(input_real, training=True)
        d_labels_real = tf.ones_like(d_logits_real)
        d_loss_real = loss_fn(y_true=d_labels_real, y_pred=d_logits_real)

    d_grads1 = d_tape1.gradient(d_loss_real, disc_model.trainable_variables)
    d_optimizer.apply_gradients(grads_and_vars=zip(d_grads1, disc_model.trainable_variables))

    with tf.GradientTape() as d_tape2:
        d_logits_fake = disc_model(g_output, training=True)
        d_labels_fake = tf.zeros_like(d_logits_fake)
        d_loss_fake = loss_fn(y_true=d_labels_fake, y_pred=d_logits_fake)

        d_loss = d_loss_fake + d_loss_real

    d_grads2 = d_tape2.gradient(d_loss_fake, disc_model.trainable_variables)
    d_optimizer.apply_gradients(grads_and_vars=zip(d_grads2, disc_model.trainable_variables))

    return g_loss, d_loss, d_loss_real, d_loss_fake, d_logits_real, d_logits_fake


def train_models(dir_path, data, checkpoint_frequency, batch_size, num_epochs, dropout, learning_rate_disc,
                 learning_rate_gen, output_image, continue_):
    if not continue_ and os.path.exists(dir_path):
        shutil.rmtree(dir_path)

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    # Create Generator #
    gen_model = make_dcgan_generator(output_size=image_size)
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_gen)
    print("###################################")
    print("Using Generator-Model:")
    gen_model.summary()

    # Create Discriminator #
    disc_model = make_dcgan_discriminator(dropout, input_size=image_size)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_disc)
    print("###################################")
    print("Using Discriminator-Model:")
    disc_model.summary()

    # init checkpoints

    ckpt = Checkpoint(
        gen_model=gen_model,
        disc_model=disc_model,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer)

    checkpoint_path = path.join(dir_path, "checkpoints")
    ckpt_manager = CheckpointManager(ckpt, checkpoint_path, max_to_keep=None)

    # if a checkpoint exists and continue is set, restore the latest checkpoint.
    if continue_ and ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).assert_existing_objects_matched()
        print('Latest checkpoint restored!!')
    else:
        print("No checkpoints were restored!!")

    # Log File #

    losses_path = path.join(dir_path, "losses.pickle")
    if path.exists(losses_path):
        with open(losses_path, mode='rb') as f:
            glob_losses = pickle.load(f)
    else:
        glob_losses = {"epoch": [], "avg_g_loss": [], "avg_d_loss": [], "d_real": [], "d_fake": []}

    # Dataset #
    train_ds = get_dataset(data, batch_size)

    # Training #

    if continue_:
        start_epoch = int(ntpath.basename(str(ckpt_manager.latest_checkpoint).split("-")[-1])) + 1
    else:
        start_epoch = 0

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    start_time = time.time()

    local_losses = {"epoch": [], "avg_g_loss": [], "avg_d_loss": [], "d_real": [], "d_fake": []}
    # nach Pyhton Machine Learning, Raschka & Mirjalili, 3rd Edition, ISBN 978-1-78995-575-0, Seite 640ff
    for epoch in range(start_epoch, num_epochs): 

        epoch_losses, epoch_d_vals = [], []

        for input_real, _ in train_ds:
            input_z = tf.random.uniform(shape=(batch_size, z_size), minval=-1.0, maxval=1.0)
            g_loss, d_loss, d_loss_real, d_loss_fake, d_logits_real, d_logits_fake = train_step(input_real, input_z, gen_model, disc_model, loss_fn, g_optimizer, d_optimizer)

            epoch_losses.append(
                (g_loss.numpy(), d_loss.numpy(), d_loss_real.numpy(), d_loss_fake.numpy())
            )

            d_probs_real = tf.reduce_mean(tf.sigmoid(d_logits_real))
            d_probs_fake = tf.reduce_mean(tf.sigmoid(d_logits_fake))

            epoch_d_vals.append((d_probs_real.numpy(), d_probs_fake.numpy()))

        avg_losses = list(np.mean(epoch_losses, axis=0))
        losses = {"epoch": epoch, "avg_g_loss": avg_losses[0], "avg_d_loss": avg_losses[1], "d_real": avg_losses[2], "d_fake": avg_losses[3]}

        for key, val in losses.items():
                    local_losses[key].append(val)

        epoch_duration = strftime('%H:%M:%S', gmtime(time.time() - start_time))
        info_text = 'Epoch {:03d} | ET {} min | Avg Losses G/D {:.4f}/{:.4f} [D-Real: {:.4f} D-Fake {:.4f}]'.format(
            epoch, epoch_duration, *list(np.mean(epoch_losses, axis=0))
        )

        print(info_text)

        number_of_preview_images = 3
        fixed_z = tf.random.uniform(shape=(number_of_preview_images, z_size), minval=-1, maxval=1)
        samples = create_samples(gen_model, fixed_z, number_of_preview_images).numpy()
        display_examples(samples, number_of_preview_images, output_image, info_text)

        ## Save model state ##
        if epoch % checkpoint_frequency == 0:
            ckpt_manager.save(checkpoint_number = epoch)

            for key, val in local_losses.items():
                    glob_value = glob_losses[key] + val
                    glob_losses[key] = glob_value

            with open(losses_path, mode='wb')as f:
                    pickle.dump(glob_losses, f)

            plot_history(dir_path, glob_losses)
            
            local_losses = {"epoch": [], "avg_g_loss": [], "avg_d_loss": [], "d_real": [], "d_fake": []}


if __name__ == '__main__':
    # Parse Arguments #
    parser = argparse.ArgumentParser(description='Train GAN to generate landscapes')
    parser.add_argument('bSize', type=int, help='Batch Size to use')
    parser.add_argument('epochs', type=int, help='Number of epochs to train')
    parser.add_argument('-c', '--checkpoints', type=int, dest="chps", default=5,
                        help='Take checkpoint every x epochs. Default = 5')
    parser.add_argument('-d', '--directory', type=str, dest="dirPath", default="training",
                        help="The output directory where the checkpoints and others are saved. It will be created if it dosen't "
                             "exist and overritten (!) if it does.")
    parser.add_argument('-x', '--data', type=str, dest="data", default="dataset",
                        help="The directory containing subdirectories (labels) with images to use for training.")
    parser.add_argument('-r', '--dropout', type=float, dest="dropout", default=0.5,
                        help="The dropout rate to use for the discriminator. Default = 0.5")
    parser.add_argument('-ld', '--learnRateDisc', type=float, dest="learnRateDisc", default=0.0002,
                        help="The learning rate for the discriminator to use. Default = 2e-4")
    parser.add_argument('-lg', '--learnRateGen', type=float, dest="learnRateGen", default=0.0002,
                        help="The learning rate for the generator to use. Default = 2e-4")
    parser.add_argument('-o', '--output', type=str, dest="output", default="live",
                        help="The name of the file to use for the live-image")
    parser.add_argument('-ct', '--continue', dest='continue_', action='store_true', default=False,
                        help="Continue training (default: Start from the beginning)")

    args = parser.parse_args()
    train_models(args.dirPath, args.data, args.chps, args.bSize, args.epochs + 1, args.dropout, args.learnRateDisc,
                 args.learnRateGen, args.output, args.continue_)
