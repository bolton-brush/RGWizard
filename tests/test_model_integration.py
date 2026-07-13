import unittest

import numpy as np

from model_manager import ModelManager
from orientation import FLIPROT_CLASSES, prepare_fliprot_prediction_image


class ModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manager = ModelManager("models")
        cls.model_info = manager.get_model(target="CEPH", task="fliprot")
        if cls.model_info is None:
            raise AssertionError("Cephalometric fliprot model was not loaded")
        cls.learner = cls.model_info["model"]

    def test_model_vocab(self):
        self.assertEqual(tuple(self.learner.dls.vocab), FLIPROT_CLASSES)

    def test_native_fastai_preprocessing_and_prediction(self):
        image = np.arange(360 * 520, dtype=np.uint32).reshape(360, 520) % 256
        prediction_image = prepare_fliprot_prediction_image(image.astype(np.uint8))

        batch = self.learner.dls.test_dl(
            [prediction_image], num_workers=0
        ).one_batch()[0]
        prediction, _, probabilities = self.learner.predict(prediction_image)

        self.assertEqual(tuple(batch.shape), (1, 3, 224, 224))
        self.assertIn(str(prediction), FLIPROT_CLASSES)
        self.assertEqual(len(probabilities), len(FLIPROT_CLASSES))


if __name__ == "__main__":
    unittest.main()
