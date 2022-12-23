import os
from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertPreTrainedModel

from zshot.config import MODELS_CACHE_PATH
from zshot.utils.file_utils import download_file

MODEL_REMOTE_URL = 'https://huggingface.co/albep/zsrc/resolve/main/zsrc'
MODEL_PATH = os.path.join(MODELS_CACHE_PATH, 'zsrc')


def load_model(device: Optional[Union[str, torch.device]] = None):
    model = ZSBert(device)
    if not os.path.isfile(MODEL_PATH):
        download_file(MODEL_REMOTE_URL, MODELS_CACHE_PATH)

    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    model.eval()
    return model


class ZSBert(BertPreTrainedModel):
    def __init__(self, device: Optional[Union[str, torch.device]] = 'cpu'):
        bertconfig = BertConfig.from_pretrained('bert-large-cased', num_labels=2, finetuning_task='fewrel-zero-shot',
                                                device=device)
        bertconfig.relation_emb_dim = 1024
        super().__init__(bertconfig)
        self.bert = BertModel(bertconfig)
        self.num_labels = 2
        self.relation_emb_dim = 1024
        self.dropout = nn.Dropout(bertconfig.hidden_dropout_prob)
        self.fclayer = nn.Linear(bertconfig.hidden_size * 3, self.relation_emb_dim)
        self.classifier = nn.Linear(
            self.relation_emb_dim, bertconfig.num_labels)
        self.batch_size = 4
        self.init_weights()
        self.bert.to(device)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        e1_mask=None,
        e2_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        # Sequence of hidden-states of the last layer.
        sequence_output = outputs[0]
        # Last layer hidden-state of the [CLS] token further processed
        pooled_output = outputs[1]
        # by a Linear layer and a Tanh activation function.

        def extract_entity(sequence_output, e_mask):
            extended_e_mask = e_mask.unsqueeze(1)
            extended_e_mask = torch.bmm(
                extended_e_mask.float(), sequence_output).squeeze(1)
            return extended_e_mask.float()

        e1_h = extract_entity(sequence_output, e1_mask)
        e2_h = extract_entity(sequence_output, e2_mask)
        context = self.dropout(pooled_output)
        pooled_output = torch.cat([context, e1_h, e2_h], dim=-1)
        pooled_output = torch.tanh(pooled_output)
        pooled_output = self.fclayer(pooled_output)
        sent_embedding = torch.tanh(pooled_output)
        sent_embedding = self.dropout(sent_embedding)

        # [batch_size x hidden_size]
        logits = self.classifier(sent_embedding).to(self.bert.device)
        # add hidden states and attention if they are here

        outputs = (torch.softmax(logits, -1),) + outputs[2:]
        if labels is not None:
            ce_loss = nn.CrossEntropyLoss()
            labels = labels.to(self.bert.device)
            loss = (ce_loss(logits.view(-1, self.num_labels), labels.view(-1)))
            outputs = (loss,) + outputs

        return outputs
