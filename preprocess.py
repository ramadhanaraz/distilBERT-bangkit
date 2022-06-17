from transformers import DistilBertTokenizer


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


class BertTokenizer(object):
    def __init__(self):
        self._title_input = None
        self._title_mask = None
        self._desc_input = None
        self._desc_mask = None
        self._max_len = 64

    def preprocess(self, data):
        self._title_input, self._title_mask = TokenizeSingle(
            data[0], self._max_len)
        self._desc_input, self._desc_mask = TokenizeSingle(
            data[1], self._max_len)

        return [self._desc_input, self._title_input, self._desc_mask, self._title_mask]
