import os
import shutil
from os import path
from time import time, strftime, gmtime

from numpy import expand_dims, prod
from numpy import mean
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras import backend
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.constraints import Constraint
from matplotlib import pyplot
import pickle


# clip model weights to a given hypercube
from tensorflow.python.keras import Input

from generator_output import plot_image


class ClipConstraint(Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}


# calculate wasserstein loss
def wasserstein_loss(y_true, y_pred):
    return backend.mean(y_true * y_pred)


# define the standalone critic model
def define_critic(in_shape=(28, 28, 1)):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # weight constraint
    const = ClipConstraint(0.01)
    # define model
    model = Sequential([
        Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=init,
               kernel_constraint=const, input_shape=in_shape),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init,
               kernel_constraint=const),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=init,
               kernel_constraint=const),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init,
               kernel_constraint=const),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=init,
               kernel_constraint=const),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        Conv2D(filters=512, kernel_size=(4, 4), strides=(2, 2), padding='same',kernel_initializer=init,
               kernel_constraint=const),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),

        Conv2D(filters=512, kernel_size=(3, 3),strides=(1, 1), padding='same', kernel_initializer=init,
               kernel_constraint=const),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),

        Flatten(),
        Dense(1)
    ])
    # compile model
    opt = RMSprop(learning_rate=0.00005)
    model.compile(loss=wasserstein_loss, optimizer=opt)
    return model


# define the standalone generator model
def define_generator(latent_dim, output_size=(72, 128, 3)):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # define model
    hidden_size = (output_size[0] // 8, output_size[1] // 8)

    model = Sequential([
        Dense(units=512 * prod(hidden_size), use_bias=False, input_dim=latent_dim),
        LeakyReLU(alpha=0.2),
        Reshape((hidden_size[0], hidden_size[1], 512)),

        Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False,
                        kernel_initializer=init),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False,
                        kernel_initializer=init),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False,
                        kernel_initializer=init),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, activation='tanh',
               kernel_initializer=init)
    ])
    return model


# define the combined generator and critic model, for updating the generator
def define_gan(generator, critic):
    # make weights in the critic not trainable
    for layer in critic.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(generator)
    # add the critic
    model.add(critic)
    # compile model
    opt = RMSprop(learning_rate=0.00005)
    model.compile(loss=wasserstein_loss, optimizer=opt)
    return model


class WGAN(object):
    def __init__(self, dataset, image_size, bach_size, critic_learn_iterations, path_like="training", load=False,
                 save_interval=20):

        self.save_interval = save_interval
        self.path = path_like
        if not load:
            if path.exists(path_like):
                shutil.rmtree(path_like)
            os.mkdir(path_like)
            os.mkdir(path.join(path_like, "g_models"))
            os.mkdir(path.join(path_like, "c_models"))
            os.mkdir(path.join(path_like, "samples"))

        self.dataset = dataset
        self.image_size = image_size
        self.bach_size = bach_size
        self.critic_learn_iterations = critic_learn_iterations
        self.latent_dim = 128

        self.loss_hist: dict = dict()

        if load:
            g_models = os.listdir(path.join(path_like, "g_models"))
            c_models = os.listdir(path.join(path_like, "c_models"))
            g_models.sort()
            c_models.sort()
            self.epoch = int(g_models[-1][-7:-3])
            self.generator_model = load_model(path.join(path_like, "g_models", g_models[-1]))
            self.critic_model = load_model(path.join(path_like, "c_models", c_models[-1]),
                                           custom_objects={"ClipConstraint": ClipConstraint,
                                                           "wasserstein_loss": wasserstein_loss})
            with open(path.join(self.path, "stats.pickle"), 'rb') as f:
                self.loss_hist = pickle.load(f)
                if not isinstance(self.loss_hist, dict):
                    self.loss_hist = {}
        else:
            self.epoch = 0
            self.generator_model = define_generator(self.latent_dim, image_size)
            self.critic_model = define_critic(image_size)

        self.generator_model.summary()
        self.critic_model.summary()
        self.gan_model = define_gan(self.generator_model, self.critic_model)

        print("Initialized WGAN SUCCESS!")

    def generate_real_samples(self, data, n_samples):
        # choose random instances
        ix = randint(0, data.shape[0], n_samples)
        # select images
        x = data[ix]
        # generate class labels, -1 for 'real'
        y = -ones((n_samples, 1))
        return x, y

    # generate points in latent space as input for the generator
    def generate_latent_points(self, n_samples):
        # generate points in the latent space
        x_input = randn(self.latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(n_samples, self.latent_dim)
        return x_input

    # use the generator to generate n fake examples, with class labels
    def generate_fake_samples(self, n_samples):
        # generate points in latent space
        x_input = self.generate_latent_points(n_samples)
        # predict outputs
        x = self.generator_model.predict(x_input)
        # create class labels with 1.0 for 'fake'
        y = ones((n_samples, 1))
        return x, y

    # generate samples and save as a plot and save the model
    def summarize_performance(self, step, hist_dict, n_samples=100):
        # prepare fake examples
        x, _ = self.generate_fake_samples(n_samples)
        # scale from [-1,1] to [0,1]
        x = (x + 1) / 2.0
        # plot images
        figure = pyplot.figure(figsize=(26, 26))
        for i in range(10 * 10):
            # define subplot
            ax = figure.add_subplot(10, 10, i + 1)
            # turn off axis
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # plot raw pixel data
            plot_image(ax, x[i])

        # save plot to file
        filename1 = 'generated_plot_%04d.jpg' % step
        pyplot.savefig(path.join(self.path, "samples", filename1))
        pyplot.close()

        with open(path.join(self.path, "stats.pickle"), 'wb') as f:
            pickle.dump(hist_dict, f)

        # remove previous model if it is not in the save interval
        if (step - 1) % self.save_interval != 0:
            prev_filename = 'model_%04d.h5' % (step-1)
            try:
                for folder in ["g_models", "c_models"]:
                    os.remove(path.join(self.path, folder, prev_filename))
            except OSError:
                print("Could not remove model with filename", prev_filename)

        # save the generator model
        filename2 = 'model_%04d.h5' % step
        self.generator_model.save(path.join(self.path, "g_models", filename2))
        self.critic_model.save(path.join(self.path, "c_models", filename2))
        print('>Saved: %s and %s' % (filename1, filename2))

    # create a line plot of loss for the gan and save to file
    def plot_history(self, d1_hist, d2_hist, g_hist):
        # plot history
        pyplot.plot(d1_hist, label='crit_real loss')
        pyplot.plot(d2_hist, label='crit_fake loss')
        pyplot.plot(g_hist, label='gen loss')
        pyplot.legend()
        pyplot.savefig(path.join(self.path, f'plot_line_plot_loss_{self.epoch}.png'))
        pyplot.close()

    def train(self, epochs):
        # calculate the size of half a batch of samples
        critic_learn_count = 0
        # lists for keeping track of loss
        self.loss_hist: dict
        c1_hist = self.loss_hist.setdefault('c1_hist', list())
        c2_hist = self.loss_hist.setdefault('c2_hist', list())
        g_hist = self.loss_hist.setdefault('g_hist', list())
        c1_tmp, c2_tmp = list(), list()
        # set start time
        start_time = time()
        epochs = epochs - self.epoch
        # manually enumerate epochs
        for i in range(epochs):
            self.epoch += 1
            current_time = time() - start_time
            print("####### Epoch", self.epoch, f"Time: {strftime('%H:%M:%S', gmtime(current_time))} #######")
            for j, batch in enumerate(self.dataset):
                # update critic model weights
                c_loss1 = self.critic_model.train_on_batch(batch, -ones((batch.shape[0], 1)))
                c1_tmp.append(c_loss1)
                # generate 'fake' examples
                x_fake, y_fake = self.generate_fake_samples(self.bach_size)
                # update critic model weights
                c_loss2 = self.critic_model.train_on_batch(x_fake, y_fake)
                c2_tmp.append(c_loss2)
                critic_learn_count += 1
                if critic_learn_count == self.critic_learn_iterations:
                    critic_learn_count = 0
                    # store critic loss
                    c1_hist.append(mean(c1_tmp))
                    c2_hist.append(mean(c2_tmp))
                    c1_tmp, c2_tmp = list(), list()
                    # prepare points in latent space as input for the generator
                    x_gan = self.generate_latent_points(self.bach_size)
                    # create inverted labels for the fake samples
                    y_gan = -ones((self.bach_size, 1))
                    # update the generator via the critic's error
                    g_loss = self.gan_model.train_on_batch(x_gan, y_gan)
                    g_hist.append(g_loss)
                    # summarize loss on this batch
                    print('\r>RealLoss=%.3f, FakeLoss=%.3f GeneratorLoss=%.3f' % (c1_hist[-1], c2_hist[-1], g_loss),
                          'Processed image', j*self.bach_size + batch.shape[0], end='', flush=True)
            # evaluate the model performance every 'epoch'
            print()
            self.summarize_performance(self.epoch, {'c1_hist': c1_hist, 'c2_hist': c2_hist, 'g_hist': g_hist})
        # line plots of loss
        self.plot_history(c1_hist, c2_hist, g_hist)



