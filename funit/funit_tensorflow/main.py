import argparse
import yaml

from trainer import Trainer


def main():
    config = get_config("../config.yaml")
    prepare_config(config)


    #trainer = Trainer(config)
    #trainer.train()


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
