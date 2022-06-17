import pickle
import os

import tensorflow as tf
import numpy as np


class dBERTPredictor(object):
    def __init__(self, model, preprocessor):
        self._model = model
        self._preprocessor = preprocessor

    def predict(self, instances, **kwargs):
        inputs = np.asarray(instances)
        preprocessed_inputs = self._preprocessor.preprocess(inputs)
        outputs = self._model.predict(preprocessed_inputs)
        index_pred = tf.math.argmax(outputs[0])
        return index_pred.numpy()

    @classmethod
    def from_path(cls, model_dir):
        model_path = os.path.join(model_dir)
        model = tf.keras.models.load_model(model_path)

        preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)

        return cls(model, preprocessor)
