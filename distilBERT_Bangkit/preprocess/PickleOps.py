"""Pickle Helper Function
  
A set of tool to save encoded labels into pickle file for further load.
  
Required dependency :
    - Pickle
"""

import pickle
import os


def toPickle(input_ids, attention_masks, labels, path):
    """Save encoded sentence, attention masks, and it's labels into pickle files.

    Args:
        input_ids (Any): Encoded sentences from the tokenizer
        attention_masks (Any): Encoded attention masks from the tokenizer
        labels (Any): Series of labels related to the encoded sentence
        path (string or os.path object): Path for pickle files to be generated
    """

    pickle_input_path = os.path.join(path, "dbert_inputs.pkl")
    pickle_mask_path = os.path.join(path, "dbert_masks.pkl")
    pickle_label_path = os.path.join(path, "dbert_labels.pkl")

    if os.path.exists(path) == False:
        os.makedirs(path)

    pickle.dump((input_ids), open(pickle_input_path, 'wb'))
    pickle.dump((attention_masks), open(pickle_mask_path, 'wb'))
    pickle.dump((labels), open(pickle_label_path, 'wb'))

    print('Saving in ', pickle_input_path, pickle_mask_path, pickle_label_path)


def fromPickle(input_path, mask_path, label_path):
    """Load up pickle file to the encoded sentence, attention masks, and it's labels.

    Args:
        input_path (string or os.path object): Path for encoded sentences pickle files to be loaded
        mask_path (string or os.path object): Path for encoded attention masks pickle files to be loaded
        label_path (string or os.path object): Path for labels pickle files to be loaded

    Returns:
        A Tensor slices: Tensor slices of encoded sentence with shape of (1, max_len)
        A Tensor slices: Tensor slices of encoded attention mask with shape of (1, max_len)
        Any: Series of labels related to the encoded sentences
    """
    input_ids = pickle.load(open(input_path, 'rb'))
    attention_masks = pickle.load(open(mask_path, 'rb'))
    labels = pickle.load(open(label_path, 'rb'))

    print('Input shape {} Attention mask shape {} Input label shape {}'.format(
        input_ids.shape, attention_masks.shape, labels.shape))

    return input_ids, attention_masks, labels
