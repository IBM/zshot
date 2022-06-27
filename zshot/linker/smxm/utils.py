import os
import zipfile
from typing import Any, Dict, List

import gdown
import torch
from transformers import BertTokenizerFast
from zshot.linker.smxm.model import BertTaggerMultiClass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SmxmInput(dict):
    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        sep_index: torch.Tensor,
        seq_mask: torch.Tensor,
        split: torch.Tensor,
        labels: torch.Tensor,
    ):
        config = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "token_type_ids": token_type_ids.to(device),
            "sep_index": sep_index.to(device),
            "seq_mask": seq_mask.to(device),
            "split": split.to(device),
            "labels": labels.to(device),
        }
        super().__init__(**config)


def load_model(url: str, output_path: str, folder_name: str) -> BertTaggerMultiClass:
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    model_file_path = os.path.join(output_path, "model.zip")
    if not os.path.isfile(model_file_path):
        gdown.download(url, output=model_file_path, quiet=False)
        with zipfile.ZipFile(model_file_path, "r") as model_zip:
            model_zip.extractall(output_path)

    model = BertTaggerMultiClass.from_pretrained(
        os.path.join(output_path, folder_name), output_hidden_states=True
    ).to(device)

    return model


def predictions_to_span_annotations(
    sentences: List[str],
    predictions: List[List[int]],
    entities: List[str],
    tokenizer: BertTokenizerFast,
) -> List[List[Dict[str, Any]]]:
    span_annotations = []

    for i, sentence in enumerate(sentences):
        sentence_span_annotations = []

        tokenization = tokenizer.encode_plus(
            sentence,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
        )

        offset_mapping = tokenization["offset_mapping"]
        mapping_input_id_to_word = tokenization.encodings[0].word_ids
        words_offset_mappings = {}
        for j, w in enumerate(mapping_input_id_to_word):
            if w in words_offset_mappings:
                words_offset_mappings[w] = (
                    words_offset_mappings[w][0],
                    offset_mapping[j][1],
                )
            elif w is not None:
                words_offset_mappings[w] = offset_mapping[j]

        for j, input_id in enumerate(mapping_input_id_to_word[:-1]):
            pred = predictions[i][j]
            if (entities[pred] != "NEG") and (input_id is not None):
                if (j > 0) and (input_id != mapping_input_id_to_word[j - 1]):
                    sentence_span_annotations.append(
                        {
                            "label": entities[pred],
                            "start": words_offset_mappings[input_id][0],
                            "end": words_offset_mappings[input_id][1],
                        }
                    )
        span_annotations.append(sentence_span_annotations)

    return span_annotations
