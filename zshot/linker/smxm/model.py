import random

import torch
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BertTaggerMultiClass(BertPreTrainedModel):
    def __init__(self, config):

        super().__init__(config)

        self.bert = BertModel(config)
        self.num_labels = config.finetuning_task["num_labels"]
        self.drop = torch.nn.Dropout(config.finetuning_task["dropout_prob"])
        self.bert_output_size = config.hidden_size
        self.linear = torch.nn.Linear(self.bert_output_size, 1)
        self.linear_zero = torch.nn.Linear(self.num_labels * self.bert_output_size, 1)
        self.linear_zero2 = torch.nn.Linear(self.bert_output_size, 1)
        self.linear_zero3 = torch.nn.Linear(self.bert_output_size, 1)
        self.softmax = torch.nn.LogSoftmax(dim=2)
        self.finetune = config.finetuning_task["transformer_finetune"]
        self.use_symbols = False
        self.class_weight = [0.01] + [1] * (self.num_labels - 1)
        self.class_weight = torch.FloatTensor(self.class_weight).to(device)
        self.description_mode = config.finetuning_task["description_mode"]
        self.filter_classes = config.finetuning_task["filter_classes"]

        self.init_weights()

    def forward(
        self,
        *args,
        input_ids,
        attention_mask,
        token_type_ids,
        sep_index,
        seq_mask,
        split,
        labels=None,
        **kwargs,
    ):
        sep_index_max = torch.max(sep_index)
        predictions = []
        predictions_zero = []
        predictions_zero_base = []
        for j in range(input_ids.size(0)):
            if j == 0:
                inp_zero = torch.stack(
                    [
                        input_ids[j][i, : sep_index_max.item()]
                        for i in range(sep_index.size(0))
                    ]
                )
                att_zero = torch.stack(
                    [
                        attention_mask[j][i, : sep_index_max.item()]
                        for i in range(sep_index.size(0))
                    ]
                )
                tok_type_zero = torch.stack(
                    [
                        token_type_ids[j][i, : sep_index_max.item()]
                        for i in range(sep_index.size(0))
                    ]
                )
                if not self.finetune:
                    with torch.no_grad():
                        words_out = self.bert(
                            input_ids=inp_zero,
                            attention_mask=att_zero,
                            token_type_ids=tok_type_zero,
                        )[0]
                else:
                    words_out = self.bert(
                        input_ids=inp_zero,
                        attention_mask=att_zero,
                        token_type_ids=tok_type_zero,
                    )[0]

                pooled_out = self.drop(words_out)
                logits = self.linear_zero3(pooled_out)
                predictions_zero_base.append(logits)
            else:
                if not self.finetune:
                    with torch.no_grad():
                        words_out = self.bert(
                            input_ids=input_ids[j],
                            attention_mask=attention_mask[j],
                            token_type_ids=token_type_ids[j],
                        )[0]
                else:
                    words_out = self.bert(
                        input_ids=input_ids[j],
                        attention_mask=attention_mask[j],
                        token_type_ids=token_type_ids[j],
                    )[0]

                words_out = torch.stack(
                    [
                        words_out[i, : sep_index_max.item(), :]
                        for i in range(words_out.size(0))
                    ]
                )
                pooled_out = self.drop(words_out)
                predictions_zero.append(self.linear_zero2(pooled_out))
                logits = self.linear(pooled_out)
                predictions.append(logits)

        random.shuffle(predictions_zero)
        predictions_zero = torch.stack(predictions_zero_base + predictions_zero)
        predictions_zero = predictions_zero.transpose(0, 1).transpose(1, 2)
        predictions_zero = predictions_zero.contiguous().view(
            predictions_zero.size(0), predictions_zero.size(1), -1
        )

        predictions_zero = torch.max(predictions_zero, dim=2)[0].unsqueeze(2)

        predictions = torch.stack(predictions)
        predictions = predictions.transpose(0, 1).transpose(1, 2).squeeze(3)

        logits = torch.cat((predictions_zero, predictions), dim=2)

        loss = 0

        if labels is not None:
            labels = torch.stack(
                [labels[i, : sep_index_max.item()] for i in range(labels.size(0))]
            )
            class_weight = [0.01] + [1.0] * (logits.size(2) - 1)
            class_weight = torch.FloatTensor(class_weight).to(device)
            weights = (
                class_weight
                if split.item() == 0
                else torch.FloatTensor([1] * logits.size(2)).to(device)
            )
            loss_fct = CrossEntropyLoss(weight=weights)
            active_logits = logits.view(-1, logits.size(2))
            active_labels = labels.view(-1)
            loss = loss_fct(active_logits, active_labels)

        return logits, loss
