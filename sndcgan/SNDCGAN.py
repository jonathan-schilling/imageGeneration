import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
import ntpath
import pathlib
import shutil

import pickle

from tensorflow.train import Checkpoint, CheckpointManager
from tensorflow.python.data import AUTOTUNE

from os import path
from time import time, strftime, gmtime

from generator_output import plot_image, create_samples


tf.random.set_seed(62)
np.random.seed(87)


def make_dcgan_generator(output_size):
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

    model._name="SNDC Generator"

    return model


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
        tf.keras.layers.Dropout(dropout_rate),

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

    model._name="SNDC Discriminator"

    return model


def get_dataset(dataset, batch_size, image_size):
    data_dir = pathlib.Path(dataset)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        seed=123,
        image_size=image_size[:2],
        batch_size=batch_size,
        crop_to_aspect_ratio=True)

    normalization_layer = tf.keras.layers.Rescaling(1. / 127.5, offset=-1)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    train_ds = train_ds.cache().shuffle(10000).prefetch(AUTOTUNE)

    return train_ds


class SNDCGAN(object):
    def __init__(self, dir_path, dataset, batch_size, dropout, learning_rate_disc,
                    learning_rate_gen, live_output, continue_, image_size, z_size):
        
        if not continue_ and os.path.exists(dir_path):
            shutil.rmtree(dir_path)

        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        self.dir_path = dir_path

        self.train_ds = get_dataset(dataset, batch_size, image_size)
        self.batch_size = batch_size
        self.z_size = z_size

        self.gen_model = make_dcgan_generator(output_size=image_size)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_gen)

        self.disc_model = make_dcgan_discriminator(dropout, input_size=image_size)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_disc)

        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.live_preview_file = path.join(dir_path, live_output + ".pdf")

        self.losses_file = path.join(dir_path, "losses.pickle")
        if path.exists(self.losses_file):
            with open(self.losses_file, mode='rb') as f:
                self.losses = pickle.load(f)
        else:
            self.losses = {"epoch": [], "avg_g_loss": [], "avg_d_loss": [], "d_real": [], "d_fake": []}
        
        ckpt = Checkpoint(
            gen_model=self.gen_model,
            disc_model=self.disc_model,
            g_optimizer=self.g_optimizer,
            d_optimizer=self.d_optimizer)

        checkpoint_path = path.join(dir_path, "checkpoints")
        self.ckpt_manager = CheckpointManager(ckpt, checkpoint_path, max_to_keep=None)


        ckpt_loaded = False
        # if a checkpoint exists and continue is set, restore the latest checkpoint.
        if continue_ and self.ckpt_manager.latest_checkpoint:
            self.start_epoch = int(ntpath.basename(str(self.ckpt_manager.latest_checkpoint).split("-")[-1])) + 1
            ckpt.restore(self.ckpt_manager.latest_checkpoint).assert_existing_objects_matched()
            ckpt_loaded = True
            print('Latest checkpoint restored!!')
        else:
            self.start_epoch = 0
            print("No checkpoints were restored!!")

        print()
        self.gen_model.summary()

        print()
        self.disc_model.summary()
        
        if ckpt_loaded:
            print("\nLatest checkpoint restored!!")
        else:
            print("\nNo checkpoints were restored!!")

        print("\nInitialized SNDCGAN successfully!\n")


    # create a line plot of loss for the gan and save to file
    def plot_history(self):
        # plot history
        plt.clf()
        for key, val in self.losses.items():
            plt.plot(val, label=key)
        plt.legend()
        plt.savefig(path.join(self.dir_path, 'plot_line_plot_loss.png'))
        plt.close()


    def display_examples(self, samples, number_of_images, info_text):
        figure = plt.figure(figsize=(20, 10))
        for j in range(number_of_images):
            ax = figure.add_subplot(1, number_of_images, j + 1)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            image = samples[j]
            plot_image(ax, image)
        figure.suptitle(info_text, size="xx-large")
        figure.savefig(self.live_preview_file)
        plt.close(figure)


    def train_step(self, input_real, input_z):
        with tf.GradientTape() as g_tape:
            g_output = self.gen_model(input_z)
            d_logits_fake = self.disc_model(g_output, training=True)
            labels_real = tf.ones_like(d_logits_fake)
            g_loss = self.loss_fn(y_true=labels_real, y_pred=d_logits_fake)

            g_grads = g_tape.gradient(g_loss, self.gen_model.trainable_variables)
            self.g_optimizer.apply_gradients(grads_and_vars=zip(g_grads, self.gen_model.trainable_variables))

        with tf.GradientTape() as d_tape1:
            d_logits_real = self.disc_model(input_real, training=True)
            d_labels_real = tf.ones_like(d_logits_real)
            d_loss_real = self.loss_fn(y_true=d_labels_real, y_pred=d_logits_real)

        d_grads1 = d_tape1.gradient(d_loss_real, self.disc_model.trainable_variables)
        self.d_optimizer.apply_gradients(grads_and_vars=zip(d_grads1, self.disc_model.trainable_variables))

        with tf.GradientTape() as d_tape2:
            d_logits_fake = self.disc_model(g_output, training=True)
            d_labels_fake = tf.zeros_like(d_logits_fake)
            d_loss_fake = self.loss_fn(y_true=d_labels_fake, y_pred=d_logits_fake)

            d_loss = d_loss_fake + d_loss_real

        d_grads2 = d_tape2.gradient(d_loss_fake, self.disc_model.trainable_variables)
        self.d_optimizer.apply_gradients(grads_and_vars=zip(d_grads2, self.disc_model.trainable_variables))

        return (g_loss, d_loss, d_loss_real, d_loss_fake, d_logits_real, d_logits_fake)


    def train(self, num_epochs, checkpoint_frequency):

        start_time = time()
        local_losses = {"epoch": [], "avg_g_loss": [], "avg_d_loss": [], "d_real": [], "d_fake": []}

        # nach Pyhton Machine Learning, Raschka & Mirjalili, 3rd Edition, ISBN 978-1-78995-575-0, Seite 640ff
        for epoch in range(self.start_epoch, num_epochs):
            epoch_losses, epoch_d_vals = [], []

            for input_real, _ in self.train_ds:
                input_z = tf.random.uniform(shape=(self.batch_size, self.z_size), minval=-1.0, maxval=1.0)
                step_output = self.train_step(input_real, input_z)

                epoch_losses.append(
                    (step_output[0].numpy(), step_output[1].numpy(), step_output[2].numpy(), step_output[3].numpy())
                )

                d_probs_real = tf.reduce_mean(tf.sigmoid(step_output[4]))
                d_probs_fake = tf.reduce_mean(tf.sigmoid(step_output[5]))

                epoch_d_vals.append((d_probs_real.numpy(), d_probs_fake.numpy()))

            avg_losses = list(np.mean(epoch_losses, axis=0))
            avg_step_loss = {"epoch": epoch, "avg_g_loss": avg_losses[0], "avg_d_loss": avg_losses[1], "d_real": avg_losses[2], "d_fake": avg_losses[3]}

            for key, val in avg_step_loss.items():
                local_losses[key].append(val)

            epoch_duration = strftime('%H:%M:%S', gmtime(time() - start_time))
            info_text = 'Epoch {:04d} | ET {} min | Avg Losses G/D {:.4f}/{:.4f} [D-Real: {:.4f} D-Fake {:.4f}]'.format(
                epoch, epoch_duration, *avg_losses)

            print(info_text)

            number_of_preview_images = 3
            fixed_z = tf.random.uniform(shape=(number_of_preview_images, self.z_size), minval=-1, maxval=1)
            samples = create_samples(self.gen_model, fixed_z, number_of_preview_images).numpy()
            self.display_examples(samples, number_of_preview_images, info_text)

            ## Save model state ##
            if epoch % checkpoint_frequency == 0:
                self.ckpt_manager.save(checkpoint_number = epoch)

                for key, val in local_losses.items():
                        glob_value = self.losses[key] + val
                        self.losses[key] = glob_value

                with open(self.losses_file, mode='wb')as f:
                        pickle.dump(self.losses, f)

                self.plot_history()
                
                local_losses = {"epoch": [], "avg_g_loss": [], "avg_d_loss": [], "d_real": [], "d_fake": []}