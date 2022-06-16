"""Tokenizer Instance
  
A set of tokenizer tool to help tokenizing the sentence required by the model before it fed.
  
Required dependency :
    - transformers module from hugginface
"""

from transformers import DistilBertTokenizer


def toTokenize(sentences, max_len):
    """Tokenize a sequence of sentences into a tensor input of ids and attention masks for training purposes.

    Args:
        sentences (list object-like or a DataFrame column): Sequence of sentences to be tokenized
        max_len (int): Maximum length of encoded sentence. Sentence shorter than this length will be padded

    Returns:
        A Tensor slices: Tensor slices of encoded sentence with shape of (n_sentence, max_len)
        A Tensor slices: Tensor slices of encoded attention masks for each sentence with shape of (n_sentence, max_len)
    """

    dbert_tokenizer = DistilBertTokenizer.from_pretrained(
        'distilbert-base-uncased')
    dbert_inputs = dbert_tokenizer(sentences.tolist(),
                                   add_special_tokens=True,
                                   max_length=max_len,
                                   pad_to_max_length=True,
                                   truncation=True,
                                   return_tensors='tf',
                                   return_attention_mask=True)

    input_ids = dbert_inputs['input_ids']
    attention_masks = dbert_inputs['attention_mask']

    return input_ids, attention_masks


def TokenizeSingle(sentence, max_len):
    """Tokenize a single sentence into a tensor input of ids and attention masks for inference purpose.

    Args:
        sentences (string): Sentence to be tokenized
        max_len (int): Maximum length of encoded sentence. Sentence shorter than this length will be padded

    Returns:
        A Tensor slices: Tensor slices of encoded sentence with shape of (1, max_len)
        A Tensor slices: Tensor slices of encoded attention mask with shape of (1, max_len)
    """

    dbert_tokenizer = DistilBertTokenizer.from_pretrained(
        'distilbert-base-uncased')
    dbert_input = dbert_tokenizer(sentence,
                                  add_special_tokens=True,
                                  max_length=max_len,
                                  pad_to_max_length=True,
                                  truncation=True,
                                  return_tensors='tf',
                                  return_attention_mask=True)

    input_id = dbert_input['input_ids']
    attention_mask = dbert_input['attention_mask']

    return input_id, attention_mask
