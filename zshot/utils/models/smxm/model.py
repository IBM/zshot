import random

import torch
from transformers import BertModel, BertPreTrainedModel, logging

logging.set_verbosity_error()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BertTaggerMultiClass(BertPreTrainedModel):
    def __init__(self, config):

        super().__init__(config)

        self.bert = BertModel(config)
        self.drop = torch.nn.Dropout(config.finetuning_task["dropout_prob"])
        self.bert_output_size = config.hidden_size
        self.linear = torch.nn.Linear(self.bert_output_size, 1)
        self.linear_zero2 = torch.nn.Linear(self.bert_output_size, 1)
        self.linear_zero3 = torch.nn.Linear(self.bert_output_size, 1)

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
                with torch.no_grad():
                    words_out = self.bert(
                        input_ids=inp_zero,
                        attention_mask=att_zero,
                        token_type_ids=tok_type_zero,
                    )[0]

                pooled_out = self.drop(words_out)
                logits = self.linear_zero3(pooled_out)
                predictions_zero_base.append(logits)
            else:
                with torch.no_grad():
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

        return logits
