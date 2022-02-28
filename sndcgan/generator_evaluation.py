#lower score means more similarity between real and generated images
import argparse
import tensorflow as tf
import os

import numpy
from numpy import mean
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import dot
from scipy.linalg import sqrtm


from generator_train import get_dataset

# Parameters #
z_size = 128


#calculate frechet inception distance
def calculate_fid(model, images_gen, images_real):
    predict_gen = model.predict(images_gen)
    predict_real = model.predict(images_real)

    mu_gen, cov_gen= mean(predict_gen,axis=0), cov(predict_gen, rowvar=False)
    mu_real, cov_real = mean(predict_real, axis=0), cov(predict_real, rowvar=False)

    ssdiff = numpy.sum((mu_gen - mu_real) ** 2.0)
    covmean = sqrtm(dot(cov_gen,cov_real))
    if iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + trace(cov_gen + cov_real - 2.0 * covmean)
    return fid

#batch_size= number of images that get evaluated
#TODO: abchecken ob model so gelanden werden kann
def evaluate_fid(checkpoints, data, batch_size, output):
    gen_checkpoint_path = checkpoints + "/generator/" + "/{epoch:04d}.ckpt"
    gen_checkpoint_dir = os.path.dirname(gen_checkpoint_path)
    gen_model = tf.keras.models.load_model(gen_checkpoint_dir + "/gen-model")

    disc_checkpoint_path = checkpoints + "/discriminator/" + "/{epoch:04d}.ckpt"
    disc_checkpoint_dir = os.path.dirname(disc_checkpoint_path)
    disc_model = tf.keras.models.load_model(disc_checkpoint_dir + "/gen-model")

    train_ds = get_dataset(data, batch_size)

    images_gen = []
    for i in batch_size:
        input_z = tf.random.uniform(shape=(batch_size, z_size), minval=-1.0, maxval=1.0)
        images_gen.append(gen_model(input_z))


    fid = calculate_fid(disc_model.layers.pop(), images_gen, train_ds[0])

    print('FID: %.3f' % fid)
    #TODO:output? öfter und dann Graph? also alle 100 epochen und verlauf?



if __name__ == '__main__':
    # Parse Arguments #
    parser = argparse.ArgumentParser(description='Train GAN to generate landscapes')
    parser.add_argument('-b', '--bSize', type=int, dest='bSize', help='Batch Size to use', default=10)
    parser.add_argument('-c', '--checkpoints', type=str, dest="checkpoints", default="training",
                        help="The output directory where the checkpoints are saved.")
    parser.add_argument('-o', '--output', type=str, dest="output", default="training",
                        help="The name of the image to (over-)write")
    parser.add_argument('-d', '--data', type=str, dest="data", default="dataset",
                        help="The directory containing subdirectories (labels) with images to use for training.")

    args = parser.parse_args()
    evaluate_fid(args.checkpoints, args.data, args.bSize, args.output)
