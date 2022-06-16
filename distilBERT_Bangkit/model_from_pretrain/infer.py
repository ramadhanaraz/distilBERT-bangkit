"""distilBERT Model Inference
  
Do an inferences for distilBERT model
  
Required dependency :
    - TensorFlow Keras
"""

import tensorflow as tf
from distilBERT_Bangkit.preprocess import Tokenizer


def predict(model, test_title, test_description, max_len):
    """Do a simple prediction from a single data.

    Args:
        model (Keras Object Model): Keras Model that has been instantiated with distilBERT pretrained layer
        test_title (string): Title input
        test_description (string): Test input
        max_len (integer): Maximum length of input (must be the same as the model required input)

    Returns:
        A numpy integer of the class related to the dataset (use a mapping dictionary to get the class names)
    """
    title_input, title_mask = Tokenizer.TokenizeSingle(test_title, max_len)
    desc_input, desc_mask = Tokenizer.TokenizeSingle(test_description, max_len)

    prediction = model.predict(
        [desc_input, title_input, desc_mask, title_mask])

    index_pred = tf.math.argmax(prediction[0])
    return index_pred.numpy()


def evaluate(model, test_input, test_label):
    """Do a model evaluation through a test dataset

    Args:
        model (Keras Object Model): Keras Model that has been instantiated with distilBERT pretrained layer
        test_input (list): A list of inputs. See model instantiation for more reference
        test_label (list or a column of DataFrame): A series of test labels

    Returns:
        A model history object: A set of evaluation done on the model
    """
    return model.evaluate(test_input, test_label)
