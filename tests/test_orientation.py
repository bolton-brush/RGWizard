import unittest

import numpy as np

from orientation import apply_fliprot_label, decode_fliprot_label


class OrientationLabelTests(unittest.TestCase):
    def test_decodes_lateral_and_frontal_labels(self):
        for view in ("lateral", "frontal"):
            for angle in ("0", "90", "180", "270"):
                for flip in ("", "F"):
                    label = f"{view}_{angle}{flip}"
                    self.assertEqual(decode_fliprot_label(label), f"{angle}{flip}")

    def test_none_requires_no_correction(self):
        image = np.arange(6).reshape(2, 3)

        changed, result = apply_fliprot_label(image, "none")

        self.assertFalse(changed)
        self.assertIs(result, image)

    def test_zero_degree_flip_is_horizontal(self):
        image = np.arange(6).reshape(2, 3)

        changed, result = apply_fliprot_label(image, "frontal_0F")

        self.assertTrue(changed)
        np.testing.assert_array_equal(result, np.fliplr(image))

    def test_rejects_unknown_labels(self):
        with self.assertRaises(ValueError):
            decode_fliprot_label("0F")


if __name__ == "__main__":
    unittest.main()
