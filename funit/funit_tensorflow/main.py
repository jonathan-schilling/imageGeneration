import argparse
import yaml

import tensorflow as tf

import matplotlib


from trainer import Trainer
from data_loader import Loader

matplotlib.use('Agg')
import matplotlib.pyplot as plt

de_normalization_layer = tf.keras.layers.Rescaling(1. / 2., offset=0.5)
def plot_image(ax, image):
    image = de_normalization_layer(image)
    ax.imshow(image)

def main():
    config = get_config("../config.yaml")
    prepare_config(config)

    # for data_content, data_class in loader:
    #     print(data_content)
    #     fig, axes = plt.subplots(figsize=(5*2, 20), nrows=len(data_content[0]), ncols=2, sharex=True, sharey=True)
    #     for i in range(len(data_content[0])):
    #         for j in range(2):
    #             ax = axes[i,j]
    #             image = data_content[0][j]
    #             ax.get_xaxis().set_visible(False)
    #             ax.get_yaxis().set_visible(False)
    #             plot_image(ax, image)
    #     break
    # fig.savefig("test.pdf")


    trainer = Trainer(config)
    trainer.train(10)


def get_config(config_file):
    with open(config_file, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def prepare_config(config: dict):
    parser = argparse.ArgumentParser(description='Train FUNITGAN to modify landscapes')
    parser.add_argument('bSize', type=int, help='Batch Size to use')
    parser.add_argument('epochs', type=int, help='Number of epochs to train')

    args = parser.parse_args()
    
    config['img_height'] = config.setdefault('img_height', 72)
    config['img_width'] = config.setdefault('img_width', 128)
    config['img_dim'] = config.setdefault('img_dim', 3)
    config['img_size'] = (config['img_height'], config['img_width'], config['img_dim'])
    config['data_dir'] = config.setdefault('data_dir', "bilderNeuro")

    config['batch_size'] = args.bSize
    config['epochs'] = args.epochs


if __name__ == '__main__':
    main()
