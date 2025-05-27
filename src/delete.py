# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a MosaicBERT wrapper around a :class:`.ComposerTransformer`."""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

# Add src folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import bert_layers as bert_layers_module
import transformers
from composer.metrics.nlp import BinaryF1Score, LanguageCrossEntropy, MaskedAccuracy
from composer.models.huggingface import HuggingFaceModel
from torchmetrics import MeanSquaredError
from torchmetrics.classification.accuracy import MulticlassAccuracy, MultilabelAccuracy
from torchmetrics.classification.matthews_corrcoef import MatthewsCorrCoef
from torchmetrics.regression.spearman import SpearmanCorrCoef

import src.bert_layers.configuration_bert as configuration_bert_module

all = ["create_mosaic_bert_mlm", "create_mosaic_bert_classification"]

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

    # def build_inputs_with_special_tokens(self, token_ids, secondary_token_ids=None):

    #     if secondary_token_ids is None:
    #         # [CLS] + tokens + [SEP]
    #         return [1] + token_ids + [2]

    #     return [1] + token_ids + [2] + secondary_token_ids + [2]

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

################# END HelixSwap ###############

def create_mosaic_bert_mlm(
    pretrained_model_name: str = "bert-base-uncased",
    model_config: Optional[dict] = None,
    tokenizer_name: Optional[str] = None,
    gradient_checkpointing: Optional[bool] = False,
    pretrained_checkpoint: Optional[str] = None,
):
    """Mosaic BERT masked language model based on |:hugging_face:| Transformers.

    For more information, see
    `Transformers. <https://huggingface.co/transformers/>`_.

    This function creates a MosaicBERT, which includes several throughput
    optimizations not available in |:hugging_face:| BERT as well as
    architecture changes based on ALiBi and Gated Linear Units.

    Args:
        pretrained_model_name (str): Name of the Hugging Face model to
            instantiate. This will determine the default model configuration.
            Default: ``bert-base-uncased``.
        model_config (dict): A dictionary of user-specified configurations to
            update/add to the default model configuration.
        tokenizer_name (str, optional): Tokenizer name used to preprocess the
            dataset and validate the models inputs.
        gradient_checkpointing (bool, optional): Use gradient checkpointing.
            Default: ``False``.
        pretrained_checkpoint (str, optional): The pretrained checkpoint to
            initialize the model weights. If provided, the state dictionary
            stored at `pretrained_checkpoint` will be loaded into the model
            after initialization. Default: ``None``.

    .. code-block::

        {
        "_name_or_path": "bert-base-uncased",
        "alibi_starting_size": 512,
        "architectures": ["BertForMaskedLM"],
        "attention_probs_dropout_prob": 0.0,
        "classifier_dropout": null,
        "gradient_checkpointing": false,
        "hidden_act": "silu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "bert",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "position_embedding_type": "absolute",
        "transformers_version": "4.16.0",
        "type_vocab_size": 2,
        "use_cache": true,
        "vocab_size": 30522
        }

    To create a MosaicBERT model for Masked Language Model pretraining:

     .. testcode::

         from src.mosaic import create_mosaic_bert_mlm
         model = create_mosaic_bert_mlm()
    """
    if not model_config:
        model_config = {}

    if not pretrained_model_name:
        pretrained_model_name = "bert-base-uncased"

    config = configuration_bert_module.BertConfig.from_pretrained(pretrained_model_name, **model_config)

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    if pretrained_checkpoint is not None:
        model = bert_layers_module.BertForMaskedLM.from_composer(
            pretrained_checkpoint=pretrained_checkpoint, config=config
        )
    else:
        model = bert_layers_module.BertForMaskedLM(config)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()  # type: ignore

    if tokenizer_name:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name)

    metrics = [
        # vocab size no longer arg in composer
        LanguageCrossEntropy(ignore_index=-100),
        MaskedAccuracy(ignore_index=-100),
    ]

    hf_model = HuggingFaceModel(
        model=model,
        tokenizer=tokenizer,
        use_logits=True,
        metrics=metrics,
        allow_embedding_resizing=model.config.allow_embedding_resizing,
    )

    # Padding for divisibility by 8
    # We have to do it again here because wrapping by HuggingFaceModel changes it
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    hf_model.model.resize_token_embeddings(config.vocab_size)

    return hf_model


def create_mosaic_bert_classification(
    num_labels: int,
    pretrained_model_name: str = "bert-base-uncased",
    model_config: Optional[dict] = None,
    tokenizer_name: Optional[str] = None,
    gradient_checkpointing: Optional[bool] = False,
    pretrained_checkpoint: Optional[str] = None,
    custom_eval_metrics: Optional[list] = [],
    multiple_choice: Optional[bool] = False,
):
    """Mosaic BERT classification model based on |:hugging_face:| Transformers.

    For more information, see `Transformers. <https://huggingface.co/transformers/>`_.

    This function creates a MosaicBERT, which includes several throughput
    optimizations not available in |:hugging_face:| BERT as well as
    architecture changes based on ALiBi and Gated Linear Units.

    Args:
        num_labels (int): The number of classes in the classification task.
        pretrained_model_name (str): Name of the Hugging Face model to
            instantiate. This will determine the default model configuration.
            Default: ``bert-base-uncased``.
        model_config (dict): A dictionary of user-specified configurations to
            update/add to the default model configuration.
        tokenizer_name (str, optional): Tokenizer name used to preprocess the
            dataset and validate the models inputs.
        gradient_checkpointing (bool, optional): Use gradient checkpointing.
            Default: ``False``.
        pretrained_checkpoint (str, optional): The pretrained checkpoint to
            initialize the model weights. If provided,
            the state dictionary stored at `pretrained_checkpoint` will be
            loaded into the model after initialization. Default: ``None``.
        custom_eval_metrics (list, optional): Classes of custom metrics to
            evaluate the model. Default: ``[]``.
        multiple_choice (bool, optional): Whether the model is used for
            multiple choice tasks. Default: ``False``.

    .. code-block::
        {
            "_name_or_path": "bert-base-uncased",
            "alibi_starting_size": 512,
            "architectures": [
            "BertForSequenceClassification
            ],
            "attention_probs_dropout_prob": 0.0,
            "classifier_dropout": null,
            "gradient_checkpointing": false,
            "hidden_act": "silu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "id2label": {
            "0": "LABEL_0",
            "1": "LABEL_1",
            "2": "LABEL_2"
            },
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "label2id": {
            "LABEL_0": 0,
            "LABEL_1": 1,
            "LABEL_2": 2
            },
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "bert",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "position_embedding_type": "absolute",
            "transformers_version": "4.16.0",
            "type_vocab_size": 2,
            "use_cache": true,
            "vocab_size": 30522
        }

    To create a MosaicBERT model for classification:

     .. testcode::
        from mosaic_bert import create_mosaic_bert_classification
        model = create_mosaic_bert_classification(num_labels=3) # if the task has three classes.

    Note:
        This function can be used to construct a BERT model for regression by
        setting ``num_labels == 1``. This will have two noteworthy effects.
        First, it will switch the training loss to :class:`~torch.nn.MSELoss`.
        Second, the returned :class:`.ComposerModel`'s train/validation metrics
        will be :class:`~torchmetrics.MeanSquaredError` and
        :class:`~torchmetrics.SpearmanCorrCoef`. For the classifcation case
        (when ``num_labels > 1``), the training loss is
        :class:`~torch.nn.CrossEntropyLoss`, and the train/validation
        metrics are :class:`~torchmetrics.MulticlassAccuracy` and
        :class:`~torchmetrics.MatthewsCorrCoef`, as well as
        :class:`.BinaryF1Score` if ``num_labels == 2``.
    """
    if not model_config:
        model_config = {}

    # By default, turn off attention dropout in MosaicBERT
    # Flash Attention 2 supports dropout in the attention module
    # while our previous Triton Flash Attention layer only works with
    # attention_probs_dropout_prob = 0.
    if "attention_probs_dropout_prob" not in model_config:
        model_config["attention_probs_dropout_prob"] = 0.0

    # Use `alibi_starting_size` to determine how large of an alibi tensor to
    # create when initializing the model. You should be able to ignore
    # this parameter in most cases.
    if "alibi_starting_size" not in model_config:
        model_config["alibi_starting_size"] = 512

    model_config["num_labels"] = num_labels

    if not pretrained_model_name:
        pretrained_model_name = "bert-base-uncased"

    model_cls = bert_layers_module.BertForSequenceClassification

    if multiple_choice:
        model_cls = bert_layers_module.BertForMultipleChoice

    config, unused_kwargs = transformers.AutoConfig.from_pretrained(
        pretrained_model_name, return_unused_kwargs=True, **model_config
    )
    # This lets us use non-standard config fields (e.g. `starting_alibi_size`)
    config.update(unused_kwargs)

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    if pretrained_checkpoint is not None:
        model = model_cls.from_composer(pretrained_checkpoint=pretrained_checkpoint, config=config)
    else:
        model = model_cls(config)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()  # type: ignore

    # setup the tokenizer
    if tokenizer_name:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name)

    if num_labels == 1:
        # Metrics for a regression model
        metrics = [MeanSquaredError(), SpearmanCorrCoef()]
    else:
        # Metrics for a classification model
        metrics = [
            MulticlassAccuracy(num_classes=num_labels, average="micro"),
            MatthewsCorrCoef(task="multiclass", num_classes=model.config.num_labels),
        ]
        if num_labels == 2:
            metrics.append(BinaryF1Score())

    if model_config.get("problem_type", "") == "multi_label_classification":
        metrics = [
            MultilabelAccuracy(num_labels=num_labels, average="micro"),
        ]
    allow_embedding_resizing = (
        model.config.allow_embedding_resizing if hasattr(model.config, "allow_embedding_resizing") else False
    )
    hf_model = HuggingFaceModel(
        model=model,
        tokenizer=tokenizer,
        use_logits=True,
        allow_embedding_resizing=allow_embedding_resizing,
        metrics=metrics,
        eval_metrics=[
            *metrics,
            *[metric_cls() for metric_cls in custom_eval_metrics],
        ],
    )

    # Padding for divisibility by 8
    # We have to do it again here because wrapping by HuggingFaceModel changes it
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    hf_model.model.resize_token_embeddings(config.vocab_size)

    return hf_model
