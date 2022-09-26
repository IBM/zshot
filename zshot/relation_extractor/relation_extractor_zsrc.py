from zshot.relation_extractor.relations_extractor import RelationsExtractor
from zshot.relation_extractor.zsrc.zero_shot_rel_class import predict, load_model
import numpy as np

from typing import Iterator
from spacy.tokens import Doc


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

    def extract_relations(self, docs: Iterator[Doc], batch_size=None):
        for doc in docs:
            items_to_process = []
            for i, e1 in enumerate(doc.ents):
                for j, e2 in enumerate(doc.ents):
                    if (
                        i == j
                        or (e1, e2) in items_to_process
                        or (e2, e1) in items_to_process
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
                            doc._.relations.append(
                                (e1, e2, p, self.relations[pred_class_idx])
                            )
