import imp
import os
from re import I
import shutil
from os import path

from numpy import expand_dims, prod
from numpy import mean
from numpy import ones
from numpy.random import randn
from numpy.random import randint
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, ReLU, \
    BatchNormalization, Layer
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.train import Checkpoint, CheckpointManager
from tensorflow_addons.layers import InstanceNormalization
import matplotlib.pyplot as plt

from data_loader import Loader


class Tanh(tf.keras.layers.Layer):
    def __init__(self):
        super(Tanh, self).__init__()

    def call(self, inputs):
        return tf.keras.activations.tanh(inputs)


class ReflectionPadding2D(Layer):  # From https://stackoverflow.com/questions/50677544/reflection-padding-conv2d
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [tf.keras.layers.InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def get_output_shape_for(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')


class ResBlock(Layer):
    def __init__(self, filters):
        super(ResBlock, self).__init__()
        self.model = Sequential([
            d_conv(filters),
            d_conv(filters)
        ])

    def call(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


def k_conv(filter, norm):
    if norm:
        model = Sequential([
            Conv2D(filters=filter, kernel_size=(4, 4), strides=(2, 2)),
            InstanceNormalization(axis=1, center=True, scale=True, beta_initializer="random_uniform",
                                  gamma_initializer="random_uniform"),
            LeakyReLU(alpha=0.2)
        ])
    else:
        model = Sequential([
            Conv2D(filters=filter, kernel_size=(4, 4), strides=(2, 2)),
            LeakyReLU(alpha=0.2)
        ])
    return model


# define the standalone critic model
def define_discriminator():
    # define model
    model = Sequential([
        k_conv(64, False),

        k_conv(128, True),

        k_conv(256, True),

        k_conv(512, True),

        Conv2D(filters=1, kernel_size=(4, 4), strides=(1, 1))
    ])

    return model


def conv_c7_s1(filters, use_tanh=False):
    model = Sequential([
        Conv2D(filters=filters, kernel_size=(7, 7), strides=(1, 1)),
        InstanceNormalization(axis=1, center=True, scale=True, beta_initializer="random_uniform",
                              gamma_initializer="random_uniform"),
        Tanh() if use_tanh else ReLU()
    ])
    return model


def d_conv(filters):
    model = Sequential([
        ReflectionPadding2D(),
        Conv2D(filters=filters, kernel_size=(3, 3), strides=(2, 2)),
        InstanceNormalization(axis=1, center=True, scale=True, beta_initializer="random_uniform",
                              gamma_initializer="random_uniform"),  # IntsanceNormalisation
        ReLU()
    ])
    return model


def u_conv(filters):
    model = Sequential([
        Conv2DTranspose(filters=filters, kernel_size=(3, 3), strides=(2, 2)),
        InstanceNormalization(axis=1, center=True, scale=True, beta_initializer="random_uniform",
                              gamma_initializer="random_uniform"),
        ReLU()
    ])
    return model


# define the standalone generator model
def define_generator():
    model = Sequential([
        conv_c7_s1(64),

        d_conv(128),
        d_conv(256),

        ResBlock(256),
        ResBlock(256),
        ResBlock(256),
        ResBlock(256),
        ResBlock(256),
        ResBlock(256),
        ResBlock(256),
        ResBlock(256),
        ResBlock(256),

        u_conv(128),
        u_conv(64),

        conv_c7_s1(3, use_tanh=True),
    ])
    return model


LAMBDA = 10
loss_fn = BinaryCrossentropy(from_logits=True)


def discriminator_loss(real, generated):
    real_loss = loss_fn(tf.ones_like(real), real)
    generated_loss = loss_fn(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss * 0.5


def generator_loss(generated):
    return loss_fn(tf.ones_like(generated), generated)


def calc_cycle_loss(real_image, cycled_image):
    loss = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return LAMBDA * loss


def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss


class CycleGAN(object):
    def __init__(self, dataset1_path, dataset2_path, path_like, batch_size, image_size):

        if not path.exists(path_like):
            os.mkdir(path_like)
        checkpoint_path = path.join(path_like, "checkpoints")
        if not path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        self.preview_output = path.join(path_like, "preview")

        self.loader = Loader(dataset1_path, dataset2_path, image_size, batch_size)
        self.batch_size = batch_size

        self.generator_g_optimizer = Adam(2e-4, beta_1=0.5)
        self.generator_f_optimizer = Adam(2e-4, beta_1=0.5)

        self.discriminator_x_optimizer = Adam(2e-4, beta_1=0.5)
        self.discriminator_y_optimizer = Adam(2e-4, beta_1=0.5)

        self.generator_g = define_generator()
        self.generator_f = define_generator()

        self.discriminator_x = define_discriminator()
        self.discriminator_y = define_discriminator()

        self.losses = {"gen_g_loss": [], "gen_f_loss": [], "identity_loss_g": [], "identity_loss_f": [],
                       "total_gen_g_loss": [], "total_gen_f_loss": [],
                       "total_cycle_loss": []}
        """
        ckpt = Checkpoint(
            self.generator_g,
            self.generator_f,
            self.discriminator_x,
            self.discriminator_y,
            self.generator_g_optimizer,
            self.generator_f_optimizer,
            self.discriminator_x_optimizer,
            self.discriminator_y_optimizer)

        ckpt_manager = CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')
        """
        print("Initialized CycleGAN SUCCESS!")

    # generate samples and save as a plot and save the model
    def summarize_performance(self, input_g, input_f, output_g, output_f, epoch_number):
        def plot_image(ax, image):
            image = de_normalization_layer(image)
            ax.imshow(image)

        def get_axis(axes, x, y):
            ax = axes[x, y]
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            return ax

        de_normalization_layer = tf.keras.layers.Rescaling(1. / 2., offset=0.5)

        number_of_cases = len(input_g) + len(input_f)

        fig, axes = plt.subplots(figsize=(10, 5 * number_of_cases), nrows=number_of_cases, ncols=2, sharex=True,
                                 sharey=True)
        first_g_ax = get_axis(axes, 0, 0)
        first_g_ax.set_title('Images for G-GAN')
        for i in range(len(input_g)):
            input_ax = get_axis(axes, 0, i)
            input_img = input_g[i]
            output_ax = get_axis(axes, 1, i)
            output_img = output_g[i]
            plot_image(input_ax, input_img)
            plot_image(output_ax, output_img)
        first_f_ax = get_axis(axes, 0, 0)
        first_f_ax.set_title('Images for F-GAN')
        n = len(input_g)
        for i in range(len(input_f)):
            n += i
            input_ax = get_axis(axes, 0, n)
            input_img = input_f[i]
            output_ax = get_axis(axes, 1, n)
            output_img = output_f[i]
            plot_image(input_ax, input_img)
            plot_image(output_ax, output_img)
        fig.suptitle(f"Batch: {epoch_number}", size='xx-large')
        fig.savefig(self.preview_output + ".pdf")

    # create a line plot of loss for the gan and save to file
    def plot_history(self):
        # plot history
        for key, val in self.losses:
            plt.plot(val, label=key)
        plt.legend()
        plt.savefig(path.join(self.path, 'plot_line_plot_loss.png'))
        plt.close()

    #@tf.function  # speed up code
    def train_step(self, real_x, real_y):
        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.

            fake_y = self.generator_g(real_x, training=True)
            cycled_x = self.generator_f(fake_y, training=True)

            fake_x = self.generator_f(real_y, training=True)
            cycled_y = self.generator_g(fake_x, training=True)

            # same_x and same_y are used for identity loss.
            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)

            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)

            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            # calculate the loss
            gen_g_loss = generator_loss(disc_fake_y)
            gen_f_loss = generator_loss(disc_fake_x)

            total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

            identity_loss_g = identity_loss(real_y, same_y)
            identity_loss_f = identity_loss(real_x, same_x)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss_g
            total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss_f
            disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss, self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss, self.generator_f.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss, self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss, self.discriminator_y.trainable_variables)

        # Apply the gradients to the optimizer
        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients, self.generator_g.trainable_variables))
        self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients, self.generator_f.trainable_variables))
        self.discriminator_x_optimizer.apply_gradients(
            zip(discriminator_x_gradients, self.discriminator_x.trainable_variables))
        self.discriminator_y_optimizer.apply_gradients(
            zip(discriminator_y_gradients, self.discriminator_y.trainable_variables))

        return {"gen_g_loss": gen_g_loss, "gen_f_loss": gen_f_loss, "identity_loss_g": identity_loss_g,
                "identity_loss_f": identity_loss_f,
                "total_gen_g_loss": total_gen_g_loss, "total_gen_f_loss": total_gen_f_loss,
                "total_cycle_loss": total_cycle_loss}

    def train(self, epochs):
        # manually enumerate epochs
        for i in range(epochs):
            print("####### Epoch", i, "#######")
            for j, (batch1, batch2) in enumerate(iter(self.loader)):
                losses = self.train_step(batch1, batch2)
                for key, val in losses.items():
                    self.losses[key].append(val)
                print(
                    f"\r>Gen losses (g/f): {losses['gen_g_loss']}/{losses['gen_f_loss']}, identity: {losses['identity_loss_g']}/{losses['identity_loss_f']},\
                     cycle: {losses['total_cycle_loss']}, total: {losses['total_gen_g_loss']}/{losses['total_gen_f_loss']}",
                    end="", flush=True)
            # evaluate the model performance every 'epoch'
            else:  # is executed after for-loop
                translated_image_g = self.generator_g(batch1[0:2], training=False)
                translated_image_f = self.generator_f(batch1[0:2], training=False)
                self.summarize_performance(batch1[0:2], batch2[0:2], translated_image_g, translated_image_f, i)
        # line plots of loss
        self.plot_history()
