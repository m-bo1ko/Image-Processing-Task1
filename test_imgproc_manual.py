import unittest
import numpy as np
import os
import tempfile
import math

from main import (
    load_image, save_image, brightness, contrast, negative,
    hflip, vflip, dflip, shrink, enlarge,
    median_filter, gmean_filter,
    add_gaussian_noise, add_salt_pepper_noise,
    mse, pmse, snr, psnr, md
)


class TestImageProcessing(unittest.TestCase):

    def setUp(self):
        self.img = np.array([
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120]]
        ], dtype=np.uint8)

    def test_brightness_increase(self):
        result = brightness(self.img, 10)
        self.assertTrue(np.array_equal(result[0, 0], [20, 30, 40]))

    def test_brightness_clipping(self):
        result = brightness(self.img, 300)
        self.assertTrue(np.all(result == 255))

    def test_contrast_equal_to_one(self):
        result = contrast(self.img, 1)
        self.assertTrue(np.array_equal(result, self.img))

    def test_negative(self):
        result = negative(self.img)
        expected = 255 - self.img
        self.assertTrue(np.array_equal(result, expected))

    def test_hflip(self):
        result = hflip(self.img)
        expected = np.array([
            [[40, 50, 60], [10, 20, 30]],
            [[100, 110, 120], [70, 80, 90]]
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(result, expected))

    def test_vflip(self):
        result = vflip(self.img)
        expected = np.array([
            [[70, 80, 90], [100, 110, 120]],
            [[10, 20, 30], [40, 50, 60]]
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(result, expected))

    def test_dflip(self):
        result = dflip(self.img)
        expected = np.array([
            [[10, 20, 30], [70, 80, 90]],
            [[40, 50, 60], [100, 110, 120]]
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(result, expected))

    def test_shrink(self):
        img = np.ones((4, 4, 3), dtype=np.uint8) * 100
        result = shrink(img, 2)
        self.assertEqual(result.shape, (2, 2, 3))

    def test_enlarge(self):
        img = np.ones((2, 2, 3), dtype=np.uint8) * 50
        result = enlarge(img, 2)
        self.assertEqual(result.shape, (4, 4, 3))
        self.assertTrue(np.all(result == 50))

    def test_median_filter(self):
        img = np.array([
            [[0, 0, 0], [255, 255, 255]],
            [[255, 255, 255], [0, 0, 0]]
        ], dtype=np.uint8)
        result = median_filter(img, 3)
        self.assertTrue(0 <= result[0, 0, 0] <= 255)

    def test_gmean_filter(self):
        img = np.ones((3, 3, 3), dtype=np.uint8) * 10
        result = gmean_filter(img, 3)
        self.assertTrue(np.allclose(result, 10, atol=1))

    def test_add_gaussian_noise(self):
        result = add_gaussian_noise(self.img, sigma=10)
        self.assertEqual(result.shape, self.img.shape)

    def test_add_salt_pepper_noise(self):
        result = add_salt_pepper_noise(self.img, prob=0.5)
        self.assertEqual(result.shape, self.img.shape)

    def test_mse_zero(self):
        val = mse(self.img, self.img)
        self.assertEqual(val, 0)

    def test_pmse_zero(self):
        val = pmse(self.img, self.img)
        self.assertEqual(val, 0)

    def test_snr_infinite(self):
        val = snr(self.img, self.img)
        self.assertTrue(math.isinf(val))

    def test_psnr_infinite(self):
        val = psnr(self.img, self.img)
        self.assertTrue(math.isinf(val))

    def test_md_zero(self):
        val = md(self.img, self.img)
        self.assertEqual(val, 0)

    def test_load_and_save_image(self):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            save_image(tmp.name, self.img)
            loaded = load_image(tmp.name)
            self.assertEqual(loaded.shape, self.img.shape)
        os.remove(tmp.name)


if __name__ == "__main__":
    unittest.main()
