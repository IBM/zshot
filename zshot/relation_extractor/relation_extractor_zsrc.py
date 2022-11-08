from zshot.relation_extractor.relations_extractor import RelationsExtractor
from zshot.relation_extractor.zsrc.zero_shot_rel_class import predict, load_model
import numpy as np
from tqdm import tqdm
from typing import Iterator, List
from spacy.tokens import Doc

from zshot.utils.data_models.relation_span import RelationSpan


class RelationsExtractorZSRC(RelationsExtractor):
    def __init__(self, thr=0.5):
        self.model = None
        self.load_models()
        self.thr = thr
        super(RelationsExtractor, self).__init__()

    def load_models(
        self,
    ):
        if self.model is None:
            self.model = load_model()

    def predict(self, docs: Iterator[Doc], batch_size=None) -> List[List[RelationSpan]]:
        relations_pred = []
        for doc in tqdm(docs, desc='classifying documents'):
            relations_doc = []
            items_to_process = []
            for i, e1 in enumerate(doc._.spans):
                for j, e2 in enumerate(doc._.spans):
                    if (
                        i == j or (e1, e2) in items_to_process or (e2, e1) in items_to_process
                    ):
                        continue
                    else:
                        items_to_process.append((e1, e2))

                    relations_probs = []
                    if self.relations is not None:
                        for rel in self.relations:
                            _, probs = predict(
                                self.model,
                                [(e1, e2, doc.text)],
                                rel.description,
                                batch_size,
                            )
                            relations_probs.append(probs[0])
                        pred_class_idx = np.argmax(np.array(relations_probs))
                        p = relations_probs[pred_class_idx]
                        if p >= self.thr:
                            relations_doc.append(
                                RelationSpan(start=e1, end=e2, score=p, relation=self.relations[pred_class_idx])
                            )
            relations_pred.append(relations_doc)
        return relations_pred
