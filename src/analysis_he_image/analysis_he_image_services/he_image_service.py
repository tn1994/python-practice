import numpy as np
from skimage.color import rgb2hed, hed2rgb

from .image_service import ImageServiceMixin


class HEImageService(ImageServiceMixin):
    """
    ref:
    https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_ihc_color_separation.html
    """

    def __init__(self):
        super(HEImageService, self).__init__()

    @staticmethod
    def convert_hed_image(image):
        # Separate the stains from the IHC image
        return rgb2hed(image)

    def convert_h_image(self, image) -> np.ndarray:
        ihc_hed = self.convert_hed_image(image=image)

        # Create an RGB image for each of the stains
        null = np.zeros_like(ihc_hed[:, :, 0])
        return hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))

    def convert_e_image(self, image) -> np.ndarray:
        ihc_hed = self.convert_hed_image(image=image)

        # Create an RGB image for each of the stains
        null = np.zeros_like(ihc_hed[:, :, 0])
        return hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
