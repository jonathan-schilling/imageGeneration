import argparse
from CycleGAN import CycleGAN

img_height, img_width = 256, 256

if __name__ == '__main__':
    # Parse Arguments #
    parser = argparse.ArgumentParser(description='Train CycleGAN to translate between image domains')
    parser.add_argument('bSize', type=int, help='Batch Size to use')
    parser.add_argument('epochs', type=int, help='Number of epochs to train')
    parser.add_argument('-x', '--data1', type=str, dest="dataset1", default="x_data",
                        help="The directory where the images from domain one can be found.")
    parser.add_argument('-y', '--data2', type=str, dest="dataset2", default="y_data",
                        help="The directory where the images from domain two can be found.")
    parser.add_argument('-d', '--directory', type=str, dest="path", default="training",
                        help="The output directory where the checkpoints are saved. It will be created if it dosen't "
                             "exist and overritten (!) if it does.")
    parser.add_argument('-c', '--checkpoints', type=int, dest="chps", default=5,
                        help='Take checkpoint every x epochs. Default = 5')
    parser.add_argument('-ct', '--continue', dest='continue_', action='store_true', default=False,
                        help="Continue training (default: Start from the beginning)")

    args = parser.parse_args()

    cycle_gan = CycleGAN(args.dataset1, args.dataset2, args.path, args.bSize, (img_width, img_height))
    cycle_gan.train(args.epochs)
