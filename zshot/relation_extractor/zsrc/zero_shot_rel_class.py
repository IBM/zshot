import os
import random
import torch
import numpy as np
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertConfig
from torch.utils.data import DataLoader
from zshot.relation_extractor.zsrc import data_helper
import urllib.request


SEED = 42
MODEL_REMOTE_URL = 'https://ibm.box.com/s/vn0betuswfxfwuor1yjg05xezty2rn8e'
MODEL_PATH = 'zshot/relation_extractor/zsrc/models/zsrc'


def get_device():
    return 'cpu'


device = get_device()
# torch.use_deterministic_algorithms(True)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

seed = 300


def get_preds(model, testloader):
    model.eval()
    preds = []
    true_labels = []
    device = get_device()
    for data in testloader:
        tokens_tensors, segments_tensors, marked_e1, marked_e2, \
            masks_tensors, label_tensor = [
                t.to(device) for t in data if t is not None]
        true_labels.extend(label_tensor.cpu().numpy())
        with torch.no_grad():
            outputs = model(input_ids=tokens_tensors,
                            token_type_ids=segments_tensors,
                            e1_mask=marked_e1,
                            e2_mask=marked_e2,
                            attention_mask=masks_tensors)
            logits = outputs[0].detach().cpu()
            preds.extend([torch.argmax(item, dim=-1) for item in logits])
    return preds, true_labels


def test(model, best_model_path, testloader):
    if model is None and best_model_path is not None:
        model = torch.load(
            best_model_path, map_location=torch.device(get_device()))
    preds, labels = get_preds(model, testloader)
    preds = [int(item.detach().cpu().numpy()) for item in preds]
    accuracy = np.mean([1 if bool(preds[i]) == bool(
        labels[i]) else 0 for i in range(len(preds))])
    return accuracy


def predict(model, items_to_process, relation_description, batch_size=4):
    trainset = data_helper.ZSDataset(
        'test', items_to_process, relation_description)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             collate_fn=data_helper.create_mini_batch_fewrel_aio, shuffle=False)
    all_preds = []
    all_probs = []
    for data in trainloader:
        tokens_tensors, segments_tensors, marked_e1, marked_e2, \
            masks_tensors, labels = [
                t.to(device) for t in data]
        if tokens_tensors.shape[1] <= 512:
            with torch.no_grad():
                outputs = model(input_ids=tokens_tensors,
                                token_type_ids=segments_tensors,
                                e1_mask=marked_e1,
                                e2_mask=marked_e2,
                                attention_mask=masks_tensors,
                                labels=labels)
                preds = outputs[1]
                probs = preds.detach().cpu().numpy()[:, 1]
                all_probs.extend(probs)
                all_preds.extend([item >= 0.5 for item in probs])
        else:
            all_probs.extend([-1] * tokens_tensors.shape[0])
            all_preds.extend([False] * tokens_tensors.shape[0])

    return all_preds, all_probs


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def download_file_to_path(source_url, dest_path):
    dest_dir = os.path.dirname(dest_path)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    urllib.request.urlretrieve(source_url, dest_path)


def load_model():
    model = ZSBert()
    if not os.path.isfile(MODEL_PATH):
        download_file_to_path(MODEL_REMOTE_URL, MODEL_PATH)

    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(get_device())
    model.eval()
    return model


class ZSBert(BertPreTrainedModel):
    def __init__(self):
        bertconfig = BertConfig.from_pretrained('bert-large-cased', num_labels=2, finetuning_task='fewrel-zero-shot')
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
        logits = self.classifier(sent_embedding).to(device)
        # add hidden states and attention if they are here

        outputs = (torch.softmax(logits, -1),) + outputs[2:]
        if labels is not None:
            ce_loss = nn.CrossEntropyLoss()
            labels = labels.to(device)
            loss = (ce_loss(logits.view(-1, self.num_labels), labels.view(-1)))
            outputs = (loss,) + outputs

        return outputs


random.seed(seed)
device = get_device()


if __name__ == '__main__':
    load_model()
