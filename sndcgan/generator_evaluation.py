#lower score means more similarity between real and generated images
import argparse
import tensorflow as tf
from os import path

import numpy
from numpy import mean
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import dot
from scipy.linalg import sqrtm


from SNDCGAN import get_dataset
from generator_output import create_samples

# Parameters #
img_height = 144
img_width = 256
image_size = (img_height, img_width, 3)
z_size = 128


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
def evaluate_fid(dir_path, dataset, batch_size, output):

    gen_model = tf.keras.models.load_model(path.join(dir_path, "models", "generator", "gen_model-20.h5"))
    disc_model = tf.keras.models.load_model(path.join(dir_path, "models", "discriminator", "disc_model-20.h5"))
    
    disc_model.pop()
    disc_model.pop()
    disc_model.add(tf.keras.layers.AveragePooling2D(pool_size=(8,8)))
    disc_model.add(tf.keras.layers.Flatten())

    print("## Models successfully loaded.")

    train_ds = get_dataset(dataset, batch_size, image_size)

    fids = []

    for i, (images_real, _) in enumerate(train_ds):

        print("## Prepare FID calculation of Batch", i)

        bSize = images_real.shape[0]

        random_z = tf.random.uniform(shape=(bSize, z_size), minval=-1.0, maxval=1.0)
        images_fake = create_samples(gen_model, random_z , bSize)

        print("## Start FID calculation of Batch", i)
        fid = calculate_fid(disc_model, images_fake, images_real)
        fids.append(fid)

    fid = mean(fids, axis=0)
    print('FID: %.3f' % fid)
    #TODO:output? öfter und dann Graph? also alle 100 epochen und verlauf?


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

    args = parser.parse_args()
    evaluate_fid(args.dirPath, args.data, args.bSize, args.output)
