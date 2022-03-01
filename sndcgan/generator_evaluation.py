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


#calculate frechet inception distance
def calculate_fid(model, images_gen, images_real):
    predict_gen = model.predict(images_gen)
    predict_real = model.predict(images_real)

    print(predict_gen.shape)
    mu_gen, cov_gen= mean(predict_gen,axis=0), cov(predict_gen, rowvar=False)
    mu_real, cov_real = mean(predict_real, axis=0), cov(predict_real, rowvar=False)

    ssdiff = numpy.sum((mu_gen - mu_real) ** 2.0)
    covmean = sqrtm(dot(cov_gen,cov_real))
    if iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + trace(cov_gen + cov_real - 2.0 * covmean)
    return fid

#batch_size= number of images that get evaluated
def evaluate_fid(dir_path, data, batch_size, output):
    gen_model = tf.keras.models.load_model(path.join(dir_path, "models", "generator", "gen_model-20.h5"))
    disc_model = tf.keras.models.load_model(path.join(dir_path, "models", "discriminator", "disc_model-20.h5"))
    disc_model.pop()
    disc_model.pop()
    disc_model.add(tf.keras.layers.AveragePooling2D())
    disc_model.add(tf.keras.layers.Flatten())
    #disc_model.summary()
    #TODO: avg pooling layer or dense layer to reduce parameters, but pooling layer not right shape
    train_ds = get_dataset(data, batch_size, image_size)
    images_real = train_ds.take(1)

    images_gen = create_samples(gen_model, tf.random.uniform(shape=(batch_size, z_size), minval=-1.0, maxval=1.0),batch_size )



    fid = calculate_fid(disc_model, images_gen, images_real)

    print('FID: %.3f' % fid)
    #TODO:output? öfter und dann Graph? also alle 100 epochen und verlauf?



if __name__ == '__main__':
    # Parse Arguments #
    parser = argparse.ArgumentParser(description='Train GAN to generate landscapes')
    parser.add_argument('-b', '--bSize', type=int, dest='bSize', help='Batch Size to use', default=10)
    parser.add_argument('-dp', '--directory', type=str, dest="dirPath", default="training",
                        help="The output directory where the checkpoints and others are saved.")
    parser.add_argument('-o', '--output', type=str, dest="output", default="training",
                        help="The name of the image to (over-)write")
    parser.add_argument('-d', '--data', type=str, dest="data", default="dataset",
                        help="The directory containing subdirectories (labels) with images to use for training.")

    args = parser.parse_args()
    evaluate_fid(args.dirPath, args.data, args.bSize, args.output)
