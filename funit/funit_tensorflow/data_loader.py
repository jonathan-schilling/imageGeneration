from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.layers import Rescaling

# img_size: (height, width, dim)
# data_dir: str
# batch_size: int

class Loader():
    def __init__(self, config) -> None:
        self.dataset = image_dataset_from_directory(
            config['data_dir'],
            seed=123,
            image_size=config['img_size'][:2],
            batch_size=config['batch_size'],
            crop_to_aspect_ratio=True)

        normalization_layer = Rescaling(1. / 127.5, offset=-1)
        self.dataset = self.dataset.map(lambda x, y: (normalization_layer(x), y))
        self.dataset = self.dataset.cache()