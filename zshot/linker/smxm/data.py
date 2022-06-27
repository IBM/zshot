from typing import Any, Dict, List, Tuple, Union

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ByDescriptionTaggerDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: Any) -> List[Dict[str, Any]]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx]


def encode_data(
    sentences: List[str],
    entity_labels: List[str],
    entity_descriptions: List[str],
    tokenizer: BertTokenizerFast,
) -> List[Dict[str, Any]]:
    encoded_data = []

    for sentence in sentences:
        tokenized_sentence = tokenizer.tokenize(sentence)
        tokenized_texts_list = []
        input_ids_list = []
        input_masks_list = []
        segment_ids_list = []

        for description in entity_descriptions:
            tokenized_description = tokenizer.tokenize(description)

            combined_tokenized_text = (
                ["[CLS]"] + tokenized_sentence + ["[SEP]"] + tokenized_description + ["[SEP]"]
            )
            split_index = combined_tokenized_text.index("[SEP]")
            input_ids = tokenizer.convert_tokens_to_ids(combined_tokenized_text)
            input_mask = [1] * len(input_ids)
            segment_ids = ([0] * split_index) + (
                [1] * (len(combined_tokenized_text) - split_index)
            )

            tokenized_texts_list.append(combined_tokenized_text)
            input_ids_list.append(input_ids)
            input_masks_list.append(input_mask)
            segment_ids_list.append(segment_ids)

        text_labels = ["NEG"] * split_index
        text_labels_ids = [entity_labels.index("NEG")] * split_index
        label_dict = {entity_labels.index("NEG"): "NEG"}

        encoded_data.append(
            {
                "text": tokenized_texts_list,
                "input_ids": input_ids_list,
                "input_masks": input_masks_list,
                "segment_ids": segment_ids_list,
                "text_labels": text_labels,
                "text_labels_ids": text_labels_ids,
                "label_dict": label_dict,
                "sep_index": split_index,
            }
        )

    return encoded_data


def tagger_multiclass_collator(
    data: Union[List[Dict[str, Any]], Dict[str, Any]]
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if isinstance(data, dict):
        data = [data]

    input_ids_lists = [f["input_ids"] for f in data]
    input_masks_lists = [f["input_masks"] for f in data]
    segment_ids_lists = [f["segment_ids"] for f in data]
    label_ids = [f["text_labels_ids"] for f in data]
    batch_size = len(data)

    sep_index = [f["sep_index"] for f in data]
    max_sep_index = max(sep_index)

    padded_bert_tokens_list = []
    padded_bert_masks_list = []
    padded_bert_segments_list = []

    padded_bert_labels = []
    padded_sequence_mask = []

    sentence_lengths = [
        max([len(bert_tokens) for bert_tokens in input_ids_lists[i]])
        for i in range(len(input_ids_lists))
    ]
    longest_sent = max(sentence_lengths)

    for i in range(batch_size):
        padded_bert_tokens = []
        padded_bert_masks = []
        padded_bert_segments = []

        padding_label = [0] * (max_sep_index - len(label_ids[i]))
        for j in range(len(input_ids_lists[i])):
            padding = [0] * (longest_sent - len(input_ids_lists[i][j]))
            padded_bert_tokens.append(input_ids_lists[i][j] + padding)
            padded_bert_masks.append(input_masks_lists[i][j] + padding)
            padded_bert_segments.append(segment_ids_lists[i][j] + padding)

        bert_tokens_t = torch.tensor(padded_bert_tokens).to(device)
        bert_masks_t = torch.tensor(padded_bert_masks).to(device)
        bert_segments_t = torch.tensor(padded_bert_segments).to(device)

        padded_bert_tokens_list.append(bert_tokens_t)
        padded_bert_masks_list.append(bert_masks_t)
        padded_bert_segments_list.append(bert_segments_t)

        padded_bert_labels.append(label_ids[i] + padding_label)
        padded_sequence_mask.append(
            sep_index[i] * [1] + (max_sep_index - sep_index[i]) * [0]
        )

    bert_tokens_t = torch.stack(padded_bert_tokens_list).transpose(1, 0)
    bert_masks_t = torch.stack(padded_bert_masks_list).transpose(1, 0)
    bert_segments_t = torch.stack(padded_bert_segments_list).transpose(1, 0)
    bert_labels_t = torch.tensor(padded_bert_labels).to(device=device, dtype=torch.long)
    bert_seq_mask_t = torch.tensor(padded_sequence_mask).to(
        device=device, dtype=torch.uint8
    )
    bert_sep_t = torch.tensor(sep_index).to(device=device)

    split = torch.tensor(2).to(device=device)

    return (
        bert_tokens_t,
        bert_masks_t,
        bert_segments_t,
        bert_sep_t,
        bert_seq_mask_t,
        split,
        bert_labels_t,
    )
