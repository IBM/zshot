import torch
from torch.utils.data import DataLoader

from zshot.relation_extractor.relations_extractor import RelationsExtractor
from zshot.relation_extractor.zsrc import data_helper
from zshot.relation_extractor.zsrc.zero_shot_rel_class import load_model
import numpy as np
from tqdm import tqdm
from typing import Iterator, List
from spacy.tokens import Doc

from zshot.utils.data_models.relation_span import RelationSpan


class RelationsExtractorZSRC(RelationsExtractor):
    def __init__(self, thr=0.5):
        super().__init__()
        self.model = None
        self.load_models()
        self.thr = thr

    def load_models(
            self,
    ):
        if self.model is None:
            self.model = load_model(self.device)

    def predict(self, docs: Iterator[Doc], batch_size=None) -> List[List[RelationSpan]]:
        relations_pred = []
        for doc in tqdm(docs, desc='classifying documents'):
            relations_doc = []
            items_to_process = []
            for i, e1 in enumerate(doc._.spans):
                for j, e2 in enumerate(doc._.spans):
                    if (
                            i == j or (e1, e2) in items_to_process or (
                            e2, e1) in items_to_process
                    ):
                        continue
                    else:
                        items_to_process.append((e1, e2))

                    relations_probs = []
                    if self.relations is not None:
                        for rel in self.relations:
                            _, probs = self._predict_internal(
                                [(e1, e2, doc.text)],
                                rel.description,
                                batch_size,
                            )
                            relations_probs.append(probs[0])
                        pred_class_idx = np.argmax(np.array(relations_probs))
                        p = relations_probs[pred_class_idx]
                        if p >= self.thr:
                            relations_doc.append(
                                RelationSpan(
                                    start=e1, end=e2, score=p, relation=self.relations[pred_class_idx])
                            )
            relations_pred.append(relations_doc)
        return relations_pred

    def _predict_internal(self, items_to_process, relation_description, batch_size=4):
        trainset = data_helper.ZSDataset(
            'test', items_to_process, relation_description)
        trainloader = DataLoader(trainset, batch_size=batch_size,
                                 collate_fn=data_helper.create_mini_batch_fewrel_aio, shuffle=False)
        all_preds = []
        all_probs = []
        for data in trainloader:
            tokens_tensors, segments_tensors, marked_e1, marked_e2, masks_tensors, labels = [
                t.to(self.device) for t in data]
            if tokens_tensors.shape[1] <= 512:
                with torch.no_grad():
                    outputs = self.model(input_ids=tokens_tensors,
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
