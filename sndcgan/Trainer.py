import argparse

from SNDCGAN import SNDCGAN

img_height = 144
img_width = 256
image_size = (img_height, img_width, 3)
z_size = 128

if __name__ == '__main__':
    # Parse Arguments #
    parser = argparse.ArgumentParser(description='GAN Trainer to generate landscape images.')
    parser.add_argument('bSize', type=int, help='Batch Size to use.')
    parser.add_argument('epochs', type=int, help='Number of epochs to train.')
    parser.add_argument('-cf', '--checkpointFrequency', type=int, dest="ckptFreq", default=5,
                        help='Take checkpoint every x epochs. Default = 5')
    parser.add_argument('-d', '--directory', type=str, dest="dirPath", default="training",
                            help="The output directory where the checkpoints and others are saved. It will be " 
                            "created if it dosen't exist and overritten (!) if it does.")
    parser.add_argument('-x', '--data', type=str, dest="data", default="dataset",
                        help="The directory containing subdirectories (labels) with images to use for training.")
    parser.add_argument('-r', '--dropout', type=float, dest="dropout", default=0.5,
                        help="The dropout rate to use for the discriminator. Default = 0.5")
    parser.add_argument('-ld', '--learnRateDisc', type=float, dest="learnRateDisc", default=0.0002,
                        help="The learning rate for the discriminator to use. Default = 2e-4")
    parser.add_argument('-lg', '--learnRateGen', type=float, dest="learnRateGen", default=0.0002,
                        help="The learning rate for the generator to use. Default = 2e-4")
    parser.add_argument('-lo', '--liveOutput', type=str, dest="liveOutput", default="live",
                        help="The name of the file to use for the live-image")
    parser.add_argument('-ct', '--continue', dest='continue_', action='store_true', default=False,
                        help="Continue training (default: Start from the beginning)")

    args = parser.parse_args()

    sndcgan = SNDCGAN(args.dirPath, args.data, args.bSize, args.dropout, args.learnRateDisc, 
                    args.learnRateGen, args.liveOutput, args.continue_, image_size, z_size)
    sndcgan.train(args.epochs + 1, args.ckptFreq)