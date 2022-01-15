from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.layers import Rescaling
from tensorflow.python.data import AUTOTUNE


class Loader:

    def __init__(self, config) -> None:
        self.config = config
        self.content_dataset = image_dataset_from_directory(
            config['data_dir'],
            seed=123,
            image_size=config['img_size'][:2],
            batch_size=config['batch_size'],
            crop_to_aspect_ratio=True,
            )

        normalization_layer = Rescaling(1. / 127.5, offset=-1)
        self.content_dataset = self.content_dataset.map(lambda x, y: (normalization_layer(x), y))
        self.content_dataset = self.content_dataset.cache().shuffle(10000).prefetch(buffer_size=AUTOTUNE)
        self.content_dataset_iter = iter(self.content_dataset)

        self.class_dataset = image_dataset_from_directory(
            config['data_dir'],
            seed=123,
            image_size=config['img_size'][:2],
            batch_size=config['batch_size'],
            crop_to_aspect_ratio=True,
        )
        self.class_dataset = self.class_dataset.map(lambda x, y: (normalization_layer(x), y))
        self.class_dataset = self.class_dataset.cache().shuffle(10000).prefetch(buffer_size=AUTOTUNE)
        self.class_dataset_iter = iter(self.class_dataset)

    def __iter__(self):
        self.content_dataset_iter = iter(self.content_dataset)
        self.class_dataset_iter = iter(self.class_dataset)
        return self

    def __next__(self):
        return next(self.content_dataset), next(self.class_dataset)