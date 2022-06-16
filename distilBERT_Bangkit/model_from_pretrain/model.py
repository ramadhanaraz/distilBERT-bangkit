"""distilBERT Model Instance
  
Initiate a distilBERT model for fine-tuning.
  
Required dependency :
    - transformers module from hugginface
    - TensorFlow Keras
    - Numpy
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate
from transformers import TFDistilBertModel, DistilBertConfig


def dBERT_model(num_classes, max_len, dropout=0.1, att_dropout=0.1, trainable=False, return_summary=True):
    """Initiate a distilBERT model with desired num_classes and shape of (None, max_len)

    Args:
        num_classes (integer): Number of classes in the dataset
        max_len (integer): Maximum length of padded encoded sentence need to be fed in
        dropout (float, optional): distilBERT configuration for hidden state dropout. Defaults to 0.1.
        att_dropout (float, optional): distilBERT configuration for self attention dropout. Defaults to 0.1.
        trainable (bool, optional): Make the pretrained layer trainable. Use this whether to finetune the upper layers. Defaults to False.
        return_summary (bool, optional): Return summary upon instantiating. Defaults to True.

    Returns:
        A Keras model object: Keras model that has been instantiated with distilBERT pretrained layer on top of it
    """
    config = DistilBertConfig(dropout=dropout,
                              attention_dropout=att_dropout,
                              output_hidden_states=True,
                              num_labels=num_classes)

    dbert_model = TFDistilBertModel.from_pretrained(
        'distilbert-base-uncased', config=config)

    for layer in dbert_model.layers:
        layer.trainable = trainable

    input_desc = Input(shape=(max_len,), dtype='int64', name='Desc_input')
    input_title = Input(shape=(max_len,), dtype='int64', name='Title_input')
    mask_desc = Input(shape=(max_len,), dtype='int64', name='Desc_mask')
    mask_title = Input(shape=(max_len,), dtype='int64', name='Title_mask')

    dbert_layer_desc = dbert_model(
        input_desc, attention_mask=mask_desc)[0][:, 0, :]
    dbert_layer_title = dbert_model(
        input_title, attention_mask=mask_title)[0][:, 0, :]

    concat_1 = Concatenate(axis=1, name='Concatenate_1')(
        [dbert_layer_desc, dbert_layer_title])
    dense_1 = Dense(512, activation='relu', name='Dense_1')(concat_1)
    dropout_1 = Dropout(0.2, name='Dropout_1')(dense_1)
    dense_2 = Dense(128, activation='relu', name='Dense_2')(dropout_1)
    dropout_2 = Dropout(0.2, name='Dropout_2')(dense_2)

    dense_last = Dense(num_classes, activation='softmax',
                       name='Output')(dropout_2)
    model = tf.keras.Model(inputs=[
                           input_desc, input_title, mask_desc, mask_title], outputs=dense_last, name='dBERT-model')
    if return_summary == True:
        print(model.summary())

    return model


def saving_checkpoint(model, path):
    """Save the model as a checkpoint for further training

    Args:
        model (Keras model object): Keras model that has been instantiated with distilBERT pretrained layer on top of it
        path (string or os object): Path to checkpoint file

    Returns:
        None
    """
    return model.save_weights(path)


def loading_checkpoint(model, path):
    """Load the model from a checkpoint for further training

    Args:
        model (Keras model object): Keras model that has been instantiated with distilBERT pretrained layer on top of it
        path (string or os object): Path to checkpoint file

    Returns:
        A Keras model object: Keras model that has been loaded with the weight from the checkpoint
    """
    return model.load_weights(path)


def savedModel(model, path):
    """Save the model in a SavedModel format

    Args:
        model (Keras model object): Keras model that has been instantiated with distilBERT pretrained layer on top of it
        path (string or os object): Path to SavedModel
    """
    return model.save(path)
