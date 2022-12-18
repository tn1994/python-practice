import io
import requests
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from skimage import data
import matplotlib.pyplot as plt

from src.analysis_he_image.analysis_he_image_services.analysis_service import AnalysisMixin
from src.analysis_he_image.analysis_he_image_services.he_image_service import HEImageService


def get_image_data() -> np.ndarray:
    """get Hematoxylin-and-eosin-stain Image as np.ndarray"""
    return np.array(Image.open('../../data/original/62995.jpeg'))


@dataclass(frozen=True)
class Data:
    image: np.ndarray = field(default_factory=get_image_data)


class Analysis(Data, HEImageService, AnalysisMixin):

    def __init__(self):
        super(Analysis, self).__init__()

    def main(self):
        try:
            gray_image: Image.Image = self.convert_gray_image(image=self.image)
            gray_ndarray: np.ndarray = self.convert_to_ndarray(
                image=gray_image)
            print(f'{gray_ndarray=}')

            self.save_image(image=self.image, save_file_name='original.jpg')

            ihc_he_binary = self.get_he_binary_image()
            print(ihc_he_binary)

            self.save_image(image=ihc_he_binary)
        except Exception as e:
            raise e

    def get_he_binary_image(self):
        ihc_h_255: np.ndarray = self.convert_h_image(image=self.image) * 255
        ihc_e_255: np.ndarray = self.convert_e_image(image=self.image) * 255
        self.save_image(image=ihc_h_255, save_file_name='ihc_h_255.jpg')
        self.save_image(image=ihc_e_255, save_file_name='ihc_e_255.jpg')

        _threshold = 200  # 128
        ihc_h_gray: Image.Image = self.convert_gray_image(ihc_h_255)
        ihc_h_binary: np.ndarray = self.thresholding(
            image=ihc_h_gray, threshold=_threshold)
        self.save_image(image=ihc_h_binary, save_file_name='ihc_h_binary.jpg')

        ihc_e_gray: Image.Image = self.convert_gray_image(ihc_e_255)
        ihc_e_binary: np.ndarray = self.thresholding(
            image=ihc_e_gray, threshold=_threshold)
        self.save_image(image=ihc_e_binary, save_file_name='ihc_e_binary.jpg')

        _condition = ((ihc_h_binary == 0) | (ihc_e_binary == 0))
        ihc_he_binary: np.ndarray = np.where(_condition, 0, 255)
        self.save_result_image(
            ihc_h_binary=ihc_h_binary,
            ihc_e_binary=ihc_e_binary,
            ihc_he_binary=ihc_he_binary)
        return ihc_he_binary

    def save_result_image(self, ihc_h_binary, ihc_e_binary, ihc_he_binary):
        """save Original, Hematoxylin Binary, Eosin Binary, Hematoxylin or Eosin Binary Image"""
        fig, axes = plt.subplots(
            2, 2, figsize=(
                14, 12), sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(self.image)
        ax[0].set_title("Original image")

        ax[1].imshow(ihc_h_binary, cmap='gray')
        ax[1].set_title("Hematoxylin")

        ax[2].imshow(ihc_e_binary, cmap='gray')
        # Note that there is no Eosin stain in this image
        ax[2].set_title("Eosin")

        ax[3].imshow(ihc_he_binary, cmap='gray')
        ax[3].set_title("Hematoxylin or Eosin")

        for a in ax.ravel():
            a.axis('off')

        fig.tight_layout()
        fig.savefig('../../data/result/result.png')


def main():
    analysis = Analysis()
    analysis.main()


if __name__ == '__main__':
    main()
