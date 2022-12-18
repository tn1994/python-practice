import pytest
import numpy as np
from numpy.testing import assert_array_equal

from src.analysis_he_image.analysis_he_image_services.analysis_service import AnalysisMixin


class TestCase:
    analysis_mixin = AnalysisMixin()

    def test_valid(self):
        expected = True
        actual = True
        assert isinstance(expected, bool)
        assert isinstance(actual, bool)
        assert expected == actual

    def test_invalid(self):
        expected = True
        actual = False
        assert isinstance(expected, bool)
        assert isinstance(actual, bool)
        with pytest.raises(AssertionError):
            assert expected == actual

    def test_valid_thresholding(self):
        expected: np.ndarray = np.ndarray([0, 128, 255])
        actual: np.ndarray = self.analysis_mixin.thresholding(image=np.ndarray([0, 128, 255]), threshold=128)
        assert isinstance(expected, np.ndarray)
        assert isinstance(actual, np.ndarray)
        assert_array_equal(expected, actual)
