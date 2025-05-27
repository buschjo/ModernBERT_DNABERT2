# HelixSwapWrapper V0.0.5
from typing import Union, List, Optional

import helix_swap_bpe as hsb
from fsspec.compression import unzip
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, TensorType, BatchEncoding
from collections import OrderedDict
import os

from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING
from transformers import PretrainedConfig
from transformers.tokenization_utils_base import TextInput, TextInputPair, PreTokenizedInput, PreTokenizedInputPair, \
    EncodedInput, EncodedInputPair, TruncationStrategy
from transformers.utils import PaddingStrategy


# This is Required for AutoLoading

class HelixSwapConfig(PretrainedConfig):
    model_type = "HelixSwapBPE"


try:
    import torch
except ImportError:
    torch = None

try:
    import numpy as np
except ImportError:
    np = None


# Beispiel:
# from transformers import AutoTokenizer
# from helixswap import HelixSwap

# tokenizer = AutoTokenizer.from_pretrained("./tokenizer_autoloader_test")

class HelixSwap(PreTrainedTokenizer):
    def __init__(self, trained_tokenizer_file, **kwargs):
        self.helix_swap_pretrained = hsb.helixswap_loader(trained_tokenizer_file)
        self._vocab = OrderedDict(sorted(self.helix_swap_pretrained.vocab().items(), key=lambda item: item[1]))
        super().__init__()
        self.add_special_tokens({'unk_token': '[UNK]',
                                 'sep_token': '[SEP]',
                                 'pad_token': '[PAD]',
                                 'cls_token': '[CLS]',
                                 'mask_token': '[MASK]'})

    def get_vocab(self):
        return self._vocab

    def _tokenize(self, x):

        return self.helix_swap_pretrained.tokenize(x)

    def _convert_id_to_token(self, ids):
        return self.helix_swap_pretrained.decode([ids])

    def vocab_size(self):
        letzter_key, lasttokenid = next(reversed(self._vocab.items()))
        return lasttokenid

    # requires new wheel of helixswapbpe
    # self.helix_swap_pretrainedvocab_size()

    def _convert_token_to_id(self, token):
        return self._vocab.get(token, 0)

    def encode(self, sequence, return_tensors=None, **kwargs):

        encoding_result = self.helix_swap_pretrained.encode(sequence)
        if return_tensors == "pt":
            return torch.tensor(encoding_result, dtype=torch.long)
        if return_tensors == "np":
            np.array(encoding_result, dtype=np.int64)
        return encoding_result

    def _encode_plus(self,
                     text: Union[TextInput, PreTokenizedInput, EncodedInput],
                     text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
                     add_special_tokens: bool = True,
                     padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
                     truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
                     max_length: Optional[int] = None,
                     stride: int = 0,
                     is_split_into_words: bool = False,
                     pad_to_multiple_of: Optional[int] = None,
                     padding_side: Optional[str] = None,
                     return_tensors: Optional[Union[str, TensorType]] = None,
                     return_token_type_ids: Optional[bool] = None,
                     return_attention_mask: Optional[bool] = None,
                     return_overflowing_tokens: bool = False,
                     return_special_tokens_mask: bool = False,
                     return_offsets_mapping: bool = False,
                     return_length: bool = False,
                     verbose: bool = True,
                     **kwargs):
        debug = False
        if debug:
            x = super()._encode_plus(text,
                                     text_pair,
                                     add_special_tokens,
                                     padding_strategy,
                                     truncation_strategy,
                                     max_length,
                                     stride,
                                     is_split_into_words,
                                     pad_to_multiple_of,
                                     padding_side,
                                     return_tensors,
                                     return_token_type_ids,
                                     return_attention_mask,
                                     return_overflowing_tokens,
                                     return_special_tokens_mask,
                                     return_offsets_mapping,
                                     return_length,
                                     verbose,
                                     **kwargs)
            print("result :", x)

        # ToDo: Potential add further cases
        def text_to_ids(text):
            if not isinstance(text, str):
                raise ValueError("Text must be a string for HelixSwap encoding plus")
            encoding = self.helix_swap_pretrained.encode(text)
            if len(encoding):

                return self.helix_swap_pretrained.encode(text)
            return []

        return self.prepare_for_model(
            text_to_ids(text),
            pair_ids=text_to_ids(text_pair) if text_pair is not None else None,
            add_special_tokens=add_special_tokens,
            padding=padding_strategy.value,
            truncation=truncation_strategy.value,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
            return_tensors=return_tensors,
            prepend_batch_axis=True,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            verbose=verbose, )

    def _batch_encode_plus(
            self,
            batch_text_or_text_pairs: Union[
                list[TextInput],
                list[TextInputPair],
                list[PreTokenizedInput],
                list[PreTokenizedInputPair],
                list[EncodedInput],
                list[EncodedInputPair],
            ],
            add_special_tokens: bool = True,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
            max_length: Optional[int] = None,
            stride: int = 0,
            is_split_into_words: bool = False,
            pad_to_multiple_of: Optional[int] = None,
            padding_side: Optional[str] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            split_special_tokens: bool = False,
            **kwargs,
    ) -> BatchEncoding:
        # Decide if we get tuple of pairs or plain strings
        if not isinstance(batch_text_or_text_pairs, (list, tuple)) and len(batch_text_or_text_pairs):
            raise ValueError("batch_text_or_text_pairs must be a list or tuple of strings or a tuple of strings")
        first = batch_text_or_text_pairs[0]
        inputids = []
        if isinstance(first, str):
            input_ids = [(x, None) for x in self.helix_swap_pretrained.encode(batch_text_or_text_pairs)]

        elif isinstance(first, tuple):
            first_elems, second_elems = list(zip(*batch_text_or_text_pairs))
            input_ids = list(zip(self.helix_swap_pretrained.encode(list(first_elems)),
                                 self.helix_swap_pretrained.encode(list(second_elems))))



        else:
            raise ValueError("first must be a string or a tuple of strings")

        batch_outputs = self._batch_prepare_for_model(
            input_ids,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            return_tensors=return_tensors,
            verbose=verbose,
            split_special_tokens=split_special_tokens,
        )
        r = BatchEncoding(batch_outputs)

        return super()._batch_encode_plus(batch_text_or_text_pairs,

                                          add_special_tokens,
                                          padding_strategy,
                                          truncation_strategy,
                                          max_length,
                                          stride,
                                          is_split_into_words,
                                          pad_to_multiple_of,
                                          padding_side,
                                          return_tensors,
                                          return_token_type_ids,
                                          return_attention_mask,
                                          return_overflowing_tokens,
                                          return_special_tokens_mask,
                                          return_offsets_mapping,
                                          return_length,
                                          verbose,
                                          split_special_tokens,
                                          **kwargs
                                          )

    def build_inputs_with_special_tokens(self, token_ids, secondary_token_ids=None):

        if secondary_token_ids is None:
            # [CLS] + tokens + [SEP]
            return [1] + token_ids + [2]

        return [1] + token_ids + [2] + secondary_token_ids + [2]

    def _save_pretrained(self, save_directory, filename_prefix=None, **kwargs):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        if filename_prefix and filename_prefix is str:
            self.helix_swap_pretrained.save(os.path.join(save_directory, "{}_vocab.json" % filename_prefix))
            return os.path.join(save_directory, "{}_vocab.json" % filename_prefix)
        self.helix_swap_pretrained.save(os.path.join(save_directory, "vocab.json"))
        return os.path.join(save_directory, "vocab.json")

    @classmethod
    def _from_pretrained(cls, configuration, tokenizer_dir, z, **kwargs):
        return cls(os.path.join(tokenizer_dir, "vocab.json"))



# Registrierung
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

CONFIG_MAPPING.register("HelixSwapBPE", HelixSwapConfig)
TOKENIZER_MAPPING.register(HelixSwapConfig, (HelixSwap, None))
