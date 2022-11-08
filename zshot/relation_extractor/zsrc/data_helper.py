# import pdb
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import BertTokenizerFast


class ZSDataset(Dataset):
    def __init__(self, mode, data, rel_desc):
        assert mode in ["train", "dev", "test"]
        self.mode = mode
        self.data = data
        self.rel_desc = rel_desc
        self.len = len(data)
        self.tokenizer = BertTokenizerFast.from_pretrained(
            "bert-base-cased", do_lower_case=False
        )

    def mark_sem_entity(
        self,
        e1_span,
        e2_span,
        sentence_token_ids,
        sent_len,
        start_of_sentence_tokens,
        sentence_tokens_positions,
    ):
        e1_start = -1
        e2_start = -1

        for i in range(0, len(sentence_tokens_positions)):
            if (e1_start == -1 and sentence_tokens_positions[i][0] >= e1_span[0] and sentence_tokens_positions[i][0] < e1_span[1]):
                e1_start = start_of_sentence_tokens + i
            elif (e2_start == -1 and sentence_tokens_positions[i][0] >= e2_span[0] and sentence_tokens_positions[i][0] < e2_span[1]):
                e2_start = start_of_sentence_tokens + i

        e1_end = 0
        e2_end = 0
        for i in range(0, len(sentence_tokens_positions)):
            if sentence_tokens_positions[i][1] == e1_span[1]:
                e1_end = (
                    start_of_sentence_tokens + i
                )  # match the last token in the span
            elif sentence_tokens_positions[i][1] == e2_span[1]:
                e2_end = (
                    start_of_sentence_tokens + i
                )  # match the last token in the span

        marked_e1 = np.array([0] * sent_len)
        marked_e2 = np.array([0] * sent_len)

        for idx in range(e1_start, min(sent_len, e1_end + 1)):
            marked_e1[idx] += 1

        for idx in range(e2_start, min(sent_len, e2_end + 1)):
            marked_e2[idx] += 1

        return torch.tensor(marked_e1, dtype=torch.long), torch.tensor(
            marked_e2, dtype=torch.long
        )

    def __getitem__(self, idx):
        g = self.data[idx]
        sentence = g[-1]

        e1_span = (g[0].start, g[0].end)
        e2_span = (g[1].start, g[1].end)
        tokens = self.tokenizer.tokenize(sentence, return_offsets_mapping=True)
        encodings = self.tokenizer(sentence, return_offsets_mapping=True)
        # tokens = encodings['input_ids']
        sentence_tokens_positions = encodings["offset_mapping"]

        relation_desc = self.rel_desc[idx]
        tokenized_relation_desc = self.tokenizer.tokenize(relation_desc)
        tokens_ids = self.tokenizer.convert_tokens_to_ids(
            ["[CLS]"] + tokenized_relation_desc + ["[SEP]"] + tokens + ["[SEP]"]
        )
        tokens_tensor = torch.tensor(tokens_ids)
        segments_tensor = torch.tensor(
            [0] * (1 + len(tokens) + 1) + [1] * (len(tokenized_relation_desc) + 1),
            dtype=torch.long,
        )
        start_of_sentence_tokens = len(["[CLS]"] + tokenized_relation_desc + ["[SEP]"])
        marked_e1, marked_e2 = self.mark_sem_entity(
            e1_span,
            e2_span,
            tokens_ids,
            len(segments_tensor),
            start_of_sentence_tokens,
            sentence_tokens_positions,
        )

        label = 0
        label_tensor = torch.tensor(label)

        return (tokens_tensor, segments_tensor, marked_e1, marked_e2, label_tensor)

    def __len__(self):
        return self.len


def create_mini_batch_fewrel_aio(samples):
    # pdb.set_trace()
    if len(samples[0]) != 5:
        samples = [samples]
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    marked_e1 = [s[2] for s in samples]
    marked_e2 = [s[3] for s in samples]
    if samples[0][4] is not None:
        label_ids = torch.stack([s[4] for s in samples])
    else:
        label_ids = None

    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)
    marked_e1 = pad_sequence(marked_e1, batch_first=True)
    marked_e2 = pad_sequence(marked_e2, batch_first=True)
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)
    return (
        tokens_tensors,
        segments_tensors,
        marked_e1,
        marked_e2,
        masks_tensors,
        label_ids,
    )
