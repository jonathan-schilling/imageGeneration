#lower score means more similarity between real and generated images
import argparse
import os

import tensorflow as tf
from os import path

import numpy
from numpy import mean
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import dot
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

from SNDCGAN import get_dataset
from generator_output import create_samples

# Parameters #
img_height = 144
img_width = 256
image_size = (img_height, img_width, 3)
z_size = 128


#form https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/ accessed on 27.02.2022
# calculate frechet inception distance
def calculate_fid(disc_model, images_fake, images_real):

    predict_fake = disc_model.predict(images_fake)
    predict_real = disc_model.predict(images_real)

    mu_fake = mean(predict_fake, axis=0)
    mu_real = mean(predict_real, axis=0)

    cov_fake = cov(predict_fake, rowvar=False)
    cov_real = cov(predict_real, rowvar=False)

    print("#### Mean and Covariances calculated.")

    ssdiff = numpy.sum((mu_fake - mu_real) ** 2.0)

    print("#### Start calculation of mean of covariances")
    covmean = sqrtm(dot(cov_fake,cov_real))
    print("#### Calculation of mean of covariances done.")

    if iscomplexobj(covmean):
        covmean = covmean.real

    print("#### Calculate Frechet Inception Distance")
    fid = ssdiff + trace(cov_fake + cov_real - 2.0 * covmean)
    return fid


# batch_size = number of images that get evaluated
# step_size = each stepsize time fid will be calculated
def evaluate_fid(dir_path, dataset, batch_size, output,step_size=1):

    list_dir_disc = [name for name in os.listdir(path.join(dir_path, "models", "discriminator"))]
    list_dir_gen  = [name for name in os.listdir(path.join(dir_path, "models", "generator"))]

    epoch_fids = []
    count = 0
    train_ds = get_dataset(dataset, batch_size, image_size)
    for file_disc in list_dir_disc:
        file_gen = "gen" + file_disc[4:]
        if (file_gen in list_dir_gen):
            count += 1
            if count % step_size != 0:
                break
            epoch = file_disc.split("-")[-1].split(".")[0]
            print("## Start loading of model "+ epoch)
            gen_model = tf.keras.models.load_model(path.join(dir_path, "models", "generator", file_gen))
            disc_model = tf.keras.models.load_model(path.join(dir_path, "models", "discriminator", file_disc))

            disc_model.pop()
            disc_model.pop()
            disc_model.add(tf.keras.layers.AveragePooling2D(pool_size=(8,8)))
            disc_model.add(tf.keras.layers.Flatten())

            print("## Models successfully loaded.")

            fids = []

            for i, (images_real, _) in enumerate(train_ds):
                print("## Prepare FID calculation of Batch", i)

                bSize = images_real.shape[0]

                random_z = tf.random.uniform(shape=(bSize, z_size), minval=-1.0, maxval=1.0)
                images_fake = create_samples(gen_model, random_z , bSize)

                print("## Start FID calculation of Batch", i)
                fid = calculate_fid(disc_model, images_fake, images_real)
                fids.append(fid)

            epoch_fids.append((epoch, mean(fids, axis=0)))

    plot_fid(epoch_fids,output)

# create a line plot of fids and save to file
def plot_fid(epoch_fids, output):
    epoch_fids = sorted(epoch_fids, key=lambda tup: int(tup[0]))
    x,y = list(zip(*epoch_fids))
    plt.clf()
    plt.plot(x,y,label="FID")
    plt.legend()
    plt.savefig(path.join(output, 'plot_line_plot_fid.png'))
    plt.close()



if __name__ == '__main__':
    # Parse Arguments #
    parser = argparse.ArgumentParser(description='Train GAN to generate landscapes')
    parser.add_argument('-b', '--bSize', type=int, dest='bSize', help='Batch Size to use', default=32)
    parser.add_argument('-d', '--directory', type=str, dest="dirPath", default="training",
                        help="The output directory where the checkpoints and others are saved.")
    parser.add_argument('-o', '--output', type=str, dest="output", default="training",
                        help="The name of the image to (over-)write")
    parser.add_argument('-x', '--data', type=str, dest="data", default="dataset",
                        help="The directory containing subdirectories (labels) with images to use for training.")
    parser.add_argument('-s', '--stepSize', type=int, dest='stepSize', help='Step Size to use for calculating FID', default=1)

    args = parser.parse_args()
    evaluate_fid(args.dirPath, args.data, args.bSize, args.output, args.stepSize)
