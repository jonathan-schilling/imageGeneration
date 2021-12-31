import tensorflow as tf
import numpy as np
import os
import PIL
import PIL.Image
import pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Parameters #

checkpoint_frequency = 5
img_height = 72
img_width = 128
batch_size = 3
data_dir = pathlib.Path("./bilderNeuro/")
num_epochs = 20
image_size = (72,128,3)
z_size = 128
mode_z = 'uniform'
selected_epochs = [0,5,10,15]

if tf.test.is_gpu_available():
  device_name = '/GPU:0'
else:
  device_name = '/CPU:0'

print(device_name)

# Generator #

def make_dcgan_generator(output_size=(72,128,3)):
  n_filters = 512
  hidden_size = (output_size[0]//8, output_size[1]//8)

  model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128,)),

    tf.keras.layers.Dense(units=512*np.prod(hidden_size), use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Reshape((hidden_size[0], hidden_size[1], 512)),

    tf.keras.layers.Conv2DTranspose(
        filters=256, kernel_size=(4,4),
        strides=(2,2), padding='same', use_bias=False
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.Conv2DTranspose(
        filters=128, kernel_size=(4,4),
        strides=(2,2), padding='same', use_bias=False
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.Conv2DTranspose(
        filters=64, kernel_size=(4,4),
        strides=(2,2), padding='same', use_bias=False
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.Conv2DTranspose(
          filters=3, kernel_size=(3,3),
          strides=(1,1), padding='same', use_bias=False,
          activation='tanh'
      )
  ])

  return model

gen = make_dcgan_generator()
gen.summary()

gen_checkpoint_path = "training_1/gen-{epoch:04d}.ckpt"
gen_checkpoint_dir = os.path.dirname(gen_checkpoint_path)

# Discriminator #

def make_dcgan_discriminator(input_size=(72,128,3)):
  model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=input_size),

    tf.keras.layers.Conv2D(
        filters=64, kernel_size=(3,3),
        strides=(1,1), padding='same'
    ),
    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.Conv2D(
        filters=128, kernel_size=(4,4),
        strides=(2,2), padding='same'
    ),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Conv2D(
        filters=128, kernel_size=(3,3),
        strides=(1,1), padding='same'
    ),
    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.Conv2D(
        filters=256, kernel_size=(4,4),
        strides=(2,2), padding='same'
    ),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Conv2D(
        filters=256, kernel_size=(3,3),
        strides=(1,1), padding='same'
    ),
    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.Conv2D(
        filters=512, kernel_size=(4,4),
        strides=(2,2), padding='same'
    ),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Conv2D(
        filters=512, kernel_size=(3,3),
        strides=(1,1), padding='same'
    ),
    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.Conv2D(
        filters=1, kernel_size=(9,16)
    ),

    tf.keras.layers.Reshape((1,))
  ])

  return model

disc_checkpoint_path = "training_1/disc-{epoch:04d}.ckpt"
disc_checkpoint_dir = os.path.dirname(disc_checkpoint_path)

disc = make_dcgan_discriminator()
disc.summary()

if mode_z == 'uniform':
  fixed_z = tf.random.uniform(shape=(batch_size, z_size), minval=-1, maxval=1)
elif mode_z == 'normal':
  fixed_z = tf.random.normal(shape=(batch_size, z_size))

def create_samples(g_model, input_z):
  g_output = g_model(input_z, training=False)
  images = tf.reshape(g_output, (batch_size, *image_size))
  return (images+1)/2.0

epoch_samples = []

with tf.device(device_name):
    gen_model = make_dcgan_generator()
    for e in selected_epochs:
        checkpoint = gen_checkpoint_path.format(epoch=e)
        gen_model.load_weights(checkpoint)

        epoch_samples.append(create_samples(gen_model, fixed_z).numpy())

de_normalization_layer = tf.keras.layers.Rescaling(1./2., offset=0.5)

fig = plt.figure(figsize=(10,14))
for i,e in enumerate(selected_epochs):
  for j in range(3):
    ax = fig.add_subplot(len(selected_epochs), 3, i*3+j+1)
    image = epoch_samples[i][j]
    image = de_normalization_layer(image)
    ax.imshow(image)
fig.savefig('test')
