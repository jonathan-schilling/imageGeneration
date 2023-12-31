#lower score means more similarity between real and generated images
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import glob
import ntpath
import argparse
import os
import shutil


import pickle

from scipy.linalg import sqrtm

from os import path

from SNDCGAN import get_dataset
from generator_output import create_samples


# Parameters #
img_height = 144
img_width = 256
image_size = (img_height, img_width, 3)
z_size = 128

max_batches = 16

tf.get_logger().setLevel('ERROR')


#form https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/ accessed on 27.02.2022
# calculate frechet inception distance
def calculate_fid(disc_model, images_fake, images_real):
    
    predict_fake = disc_model.predict(images_fake)
    predict_real = disc_model.predict(images_real)

    mu_fake = np.mean(predict_fake, axis=0)
    mu_real = np.mean(predict_real, axis=0)

    cov_fake = np.cov(predict_fake, rowvar=False)
    cov_real = np.cov(predict_real, rowvar=False)
    
    ssdiff = np.sum((mu_fake - mu_real) ** 2.0)

    covmean = sqrtm(np.dot(cov_fake,cov_real))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = ssdiff + np.trace(cov_fake + cov_real - 2.0 * covmean)
    return fid


def init_fid_evaluation(dir_path, dataset, batch_size, output, step_size, start_epoch, disc_epoch):
    
    tmp_init_file = path.join(output, "fid_tmp_init.pickle")
    tmp_init_dict = dict()
    
    model_path_gen = path.join(dir_path, "models", "generator")
    
    mdls = glob.glob(path.join(model_path_gen, "*.h5"))   
    mdls_existing = [int(ntpath.basename(y).split(".")[-2].replace("gen_model-","")) for y in mdls]
    mdls_existing.sort()
    epochs_used = [x for x in mdls_existing if x >= start_epoch]
    epochs_used = epochs_used[::step_size]

    train_ds = get_dataset(dataset, batch_size, image_size)
    
    img_real_used = []
    random_z_used = []
    batches_used = max_batches
    
    if len(train_ds) < max_batches:
        batches_used = len(train_ds)
        
    for i, (images_real, _) in enumerate(train_ds):
        
            bSize = images_real.shape[0]
            random_z = tf.random.uniform(shape=(bSize, z_size), minval=-1.0, maxval=1.0)

            img_real_used.append(images_real)
            random_z_used.append(random_z)
            
            if i == batches_used-1:
                break
    
    tmp_init_dict = {
        "epochs_used": epochs_used,
        "img_real_used": img_real_used,
        "random_z_used": random_z_used,
        "batches_used": batches_used,
        "disc_epoch": disc_epoch
    } 
    
    with open(tmp_init_file, mode='wb')as f:
        pickle.dump(tmp_init_dict, f)
        
    return tmp_init_dict


# batch_size = number of images that get evaluated
# step_size = each stepsize time fid will be calculated
def evaluate_fid(dir_path, dataset, batch_size, output, step_size, start_epoch, disc_epoch, continue_):
    
    output = path.join(output, "evaluation")
    
    if not continue_ and os.path.exists(output):
            shutil.rmtree(output)

    if not os.path.exists(output):
            os.mkdir(output)
    
    if not continue_:
        tmp_init_dict = init_fid_evaluation(dir_path, dataset, batch_size, output, step_size, start_epoch, disc_epoch)
    else:
        tmp_init_file = path.join(output, "fid_tmp_init.pickle")
        tmp_init_dict = dict()
        with open(tmp_init_file, mode='rb') as f:
            tmp_init_dict = pickle.load(f)
            
    model_path_disc = path.join(dir_path, "models", "discriminator")
    model_path_gen = path.join(dir_path, "models", "generator")
    
    epochs_used = tmp_init_dict.get("epochs_used")
    img_real_used = tmp_init_dict.get("img_real_used")
    random_z_used = tmp_init_dict.get("random_z_used")
    batches_used = tmp_init_dict.get("batches_used")
    disc_epoch = tmp_init_dict.get("disc_epoch")
    
    disc_model = tf.keras.models.load_model(path.join(model_path_disc, "disc_model-"+str(disc_epoch)+".h5"))
    disc_model.pop()
    disc_model.pop()
    disc_model.add(tf.keras.layers.AveragePooling2D(pool_size=(8,8)))
    disc_model.add(tf.keras.layers.Flatten())

    
    tmp_results_file = path.join(output, "fid_tmp_results.pickle")
    
    if not continue_ or not path.exists(tmp_results_file):
        epochs_finished = []
        epoch_fids = []
    else:
        tmp_results_dict = dict()
        
        with open(tmp_results_file, mode='rb') as f:
            tmp_results_dict = pickle.load(f)

        epochs_finished = tmp_results_dict.get("epochs_finished")
        epoch_fids = tmp_results_dict.get("epoch_fids")
    
    epochs_todo = [eps for eps in epochs_used if (eps not in epochs_finished)]
    
    for i, model in enumerate(epochs_todo):
        print("\n## Start FID calculation of epoch", model)
        
        gen_model = tf.keras.models.load_model(path.join(model_path_gen, "gen_model-"+str(model)+".h5"))
        
        fids = []
        
        for i, images_real in enumerate(img_real_used):
            print(f"\r### Calculate FID of Batch {i+1}/{batches_used}", end="", flush=True)
            
            images_fake = create_samples(gen_model, random_z_used[i] , images_real.shape[0])

            fid = calculate_fid(disc_model, images_fake, images_real)
            fids.append(fid)
            
            if i == batches_used-1:
                break

        epochs_finished.append(model)
        epoch_fids.append(fids)
        
        tmp_results_dict = {
            "epochs_finished": epochs_finished,
            "epoch_fids": epoch_fids
        } 

        with open(tmp_results_file, mode='wb')as f:
            pickle.dump(tmp_results_dict, f)
        
    print("\n\n## Calculation finished.")
    
    results_dict = {
        "epochs": epochs_used,
        "fids": epoch_fids
    } 
    results_file = path.join(output, "fid_results.pickle") 
    with open(results_file, mode='wb')as f:
        pickle.dump(results_dict, f)
        
    plot_fid_advc(epochs_used, epoch_fids, output)
    plot_fid(epochs_used, epoch_fids, output)
    
    return (epochs_used, epoch_fids)


def plot_fid_advc(epochs_used, epoch_fids, output):
    plt.clf()

    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(3*len(epochs_used), 12))

    ax.boxplot(epoch_fids,
                vert=True,
                showmeans=True,
                meanline=True,
                labels=epochs_used)

    ax.yaxis.grid(True)
    ax.set_yscale('log')
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Fréchet inception distance', fontsize=14)
    
    plt.plot([], [], '--', linewidth=1, color='tab:green', label='mean')
    plt.plot([], [], '-', linewidth=1, color='tab:orange', label='median')
    plt.plot([], [], 'o', linewidth=1, color='k', label='outlier', fillstyle='none')
    plt.legend()
    
    plt.tight_layout()
        
    plt.savefig(path.join(output, 'plot_boxplot_fids.pdf'), dpi=300)
    plt.close()


def plot_fid(epochs_used, epoch_fids, output):
    plt.clf()
    
    plt.plot(epochs_used, np.median(epoch_fids, axis=1), label='median')
    plt.plot(epochs_used, np.mean(epoch_fids, axis=1), label='mean')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Fréchet inception distance', fontsize=12)
    plt.legend()
    
    plt.yscale('log')
    plt.xticks(epochs_used)
    plt.tight_layout()
    
    plt.savefig(path.join(output, 'plot_line_plot_fids.pdf'), dpi=300)
    plt.close()


if __name__ == '__main__':
    # Parse Arguments #
    parser = argparse.ArgumentParser(description='Train GAN to generate landscapes')
    parser.add_argument('discEpoch', type=int, help='Epoch of discriminator that should be used for FID calculation.')
    parser.add_argument('-b', '--bSize', type=int, dest='bSize', help='Batch Size of images that are used to calculate the FID. Default = 32', default=32)
    parser.add_argument('-d', '--directory', type=str, dest="dirPath", default="training",
                        help="The output directory where the checkpoints and others are saved.")
    parser.add_argument('-o', '--output', type=str, dest="output", default="training",
                        help="The name of the image to (over-)write")
    parser.add_argument('-x', '--data', type=str, dest="data", default="dataset",
                        help="The directory containing subdirectories (labels) with images to use for training.")
    parser.add_argument('-st', '--stepSize', type=int, dest='stepSize', help='Calculate FID for every xth checkpoint', default=1)
    parser.add_argument('-se', '--start', type=int, dest="start", default=1, help="Start at this epoch")
    parser.add_argument('-ct', '--continue', dest='continue_', action='store_true', default=False,
                        help="Continue evaluation (default: Start from the beginning)")

    args = parser.parse_args()
    evaluate_fid(args.dirPath, args.data, args.bSize, args.output, args.stepSize, args.start, args.discEpoch, args.continue_)
