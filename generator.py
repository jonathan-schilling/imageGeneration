import tensorflow as tf
import numpy as np
import os
import PIL
import PIL.Image
import pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if tf.test.is_gpu_available():
  device_name = tf.test_gpu_device_name()
else:
  device_name = '/CPU:0'

print(device_name)

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

gen = make_dcgan_generator()
gen.summary()

disc = make_dcgan_discriminator()
disc.summary()

z = tf.random.uniform(shape=(1,128), minval=-1, maxval=1)
x = gen(z, training=False)
disc(x, training=False)


img_height = 72
img_width = 128
batch_size = 3

data_dir = pathlib.Path("./bilderNeuro/")

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  seed=123,
  image_size=(img_height,img_width),
  batch_size=batch_size,
  crop_to_aspect_ratio=True)

# def preprocess(ex, mode='uniform'):
#   image = ex['image']
#   image = tf.image.convert_image_dtype(image, tf.float32)
#   
#   image = image*2 - 1.0
#   if mode == 'uniform':
#     input_z = tf.random.uniform(shape=(z_size,), minval=-1.0, maxval=1.0)
#   elif mode == 'normal':
#     input_z = tf.random.normal(shape=(z_size,))
#   return input_z, image

import time

num_epochs = 20
batch_size = 3
image_size = (72,128,3)
z_size = 128
mode_z = 'uniform'

tf.random.set_seed(1)
np.random.seed(1)

if mode_z == 'uniform':
  fixed_z = tf.random.uniform(shape=(batch_size, z_size), minval=-1, maxval=1)
elif mode_z == 'normal':
  fixed_z = tf.random.normal(shape=(batch_size, z_size))

def create_samples(g_model, input_z):
  g_output = g_model(input_z, training=False)
  images = tf.reshape(g_output, (batch_size, *image_size))
  return (images+1)/2.0

#train_ds = train_ds.map(lambda ex: preprocess(ex, mode=mode_z))
normalization_layer = tf.keras.layers.Rescaling(1./127.5, offset=-1)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
train_ds = train_ds.shuffle(60)

with tf.device(device_name):
  gen_model = make_dcgan_generator()

  disc_model = make_dcgan_discriminator()

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
g_optimizer = tf.keras.optimizers.Adam()
d_optimizer = tf.keras.optimizers.Adam()

all_losses = []
all_d_vals = []
epoch_samples = []
start_time = time.time()
for epoch in range(1,num_epochs+1):

  epoch_losses, epoch_d_vals = [], []

  for input_real, _ in train_ds:
        input_z = tf.random.uniform(shape=(batch_size,z_size), minval=-1.0, maxval=1.0)
        with tf.GradientTape() as g_tape:
          g_output = gen_model(input_z)
          d_logits_fake = disc_model(g_output, training=True)
          labels_real = tf.ones_like(d_logits_fake)
          g_loss = loss_fn(y_true=labels_real,y_pred=d_logits_fake)
        
        g_grads = g_tape.gradient(g_loss, gen_model.trainable_variables)
        g_optimizer.apply_gradients(grads_and_vars=zip(g_grads,gen_model.trainable_variables))
        
        with tf.GradientTape() as d_tape:
          d_logits_real = disc_model(input_real,training=True)
          d_labels_real = tf.ones_like(d_logits_real)
          d_loss_real = loss_fn(y_true=d_labels_real,y_pred=d_logits_real)
          
          d_logits_fake = disc_model(g_output,training=True)
          d_labels_fake = tf.zeros_like(d_logits_fake)
          d_loss_fake = loss_fn(y_true=d_labels_fake,y_pred=d_logits_fake)
        
          d_loss = d_loss_real + d_loss_fake
        
        if d_loss > 0.001:
            d_grads = d_tape.gradient(d_loss,disc_model.trainable_variables)
            d_optimizer.apply_gradients(grads_and_vars=zip(d_grads,disc_model.trainable_variables))
        
        epoch_losses.append(
            (g_loss.numpy(), d_loss.numpy(), d_loss_real.numpy(), d_loss_fake.numpy())
        )
        
        d_probs_real = tf.reduce_mean(tf.sigmoid(d_logits_real))
        d_probs_fake = tf.reduce_mean(tf.sigmoid(d_logits_fake))
        
        epoch_d_vals.append((d_probs_real.numpy(),d_probs_fake.numpy()))

  all_losses.append(epoch_losses)
  all_d_vals.append(epoch_d_vals)

  print(
      'Epoch {:03d} | ET {:.2f} min | Avg Losses G/D {:.4f}/{:.4f} [D-Real: {:.4f} D-Fake {:.4f}]'.format(
          epoch, (time.time() - start_time)/60, *list(np.mean(all_losses[-1], axis=0))
      )
  )

  epoch_samples.append(create_samples(gen_model, fixed_z).numpy())

selected_epochs = [1,3,5,8,10]

fig = plt.figure(figsize=(10,14))
for i,e in enumerate(selected_epochs):
  for j in range(3):
    ax = fig.add_subplot(len(selected_epochs), 3, i*3+j+1)
    image = epoch_samples[e-1][j]
    ax.imshow(image)
fig.savefig('test')
