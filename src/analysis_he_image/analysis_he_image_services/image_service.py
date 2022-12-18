import os
from typing import Final

import numpy as np
from PIL import Image


class ImageServiceMixin:

    @staticmethod
    def convert_gray_image(image) -> Image.Image:
        _mode: Final[str] = 'L'
        match type(image):
            case Image.Image:
                return image.convert(mode=_mode)
            case np.ndarray:
                return Image.fromarray(
                    image.astype(
                        np.uint8)).convert(
                    mode=_mode)
            case _:
                print(f'{type(image)=}')

    @staticmethod
    def convert_to_ndarray(image: Image.Image) -> np.ndarray:
        try:
            if not isinstance(image, Image.Image):
                raise TypeError
            return np.array(image)
        except Exception as e:
            raise e

    @staticmethod
    def save_image(
            image,
            save_file_name: str = 'tmp.jpg',
            quality: int = 100) -> None:
        _dir_path: Final[str] = '../../data/result'
        _save_path: Final[str] = os.path.join(_dir_path, save_file_name)
        match type(image):
            case Image.Image:
                image.save(_save_path, quality=quality)
            case np.ndarray:
                Image.fromarray(
                    image.astype(
                        np.uint8)).save(
                    _save_path,
                    quality=quality)
            case _:
                print(f'{type(image)=}')
