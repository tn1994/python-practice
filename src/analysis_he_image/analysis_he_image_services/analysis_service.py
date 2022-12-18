import numpy as np
from PIL import Image


class AnalysisMixin:

    @staticmethod
    def thresholding(image: np.ndarray | Image.Image, threshold: int):
        match type(image):
            case Image.Image:
                return np.where(np.array(image) <= threshold, 0, 255)
            case np.ndarray:
                return np.where(image <= threshold, 0, 255)
