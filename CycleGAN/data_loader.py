from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.layers import Rescaling
from tensorflow.python.data import AUTOTUNE

class Loader:

    def __init__(self, config) -> None:
        self.config = config
        self.x_dataset = self.create_dataset(self.config['x_data_dir'])
        self.y_dataset = self.create_dataset(self.config['y_data_dir'])
        self.batch_size = self.config['batch_size']

    def create_dataset(self, data_dir):
        dataset = image_dataset_from_directory(
            data_dir,
            seed=123,
            image_size=(self.config['img_height'],self.config['img_height']),
            batch_size=self.config['batch_size'],
            crop_to_aspect_ratio=True,
            )

        normalization_layer = Rescaling(1. / 127.5, offset=-1)
        dataset = dataset.map(lambda x, y: (normalization_layer(x), y))
        dataset = dataset.cache().shuffle(10000).prefetch(buffer_size=AUTOTUNE)
        return dataset

    def __iter__(self):
        self.x_dataset_iter = iter(self.x_dataset)
        self.y_dataset_iter = iter(self.y_dataset)
        return self

    def __next__(self):
        def get_batch_size(b):
            return b[0].shape[0]
        x_next = next(self.x_dataset_iter)
        y_next = next(self.y_dataset_iter)
        # Both batches must be exactly batch_size big:
        if (get_batch_size(x_next) != self.batch_size or
                get_batch_size(y_next) != self.batch_size):
            return self.__next__()
        else:
            return x_next, y_next
