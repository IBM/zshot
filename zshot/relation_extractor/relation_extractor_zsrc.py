from zshot.relation_extractor.relations_extractor import RelationsExtractor
from zshot.relation_extractor.zsrc.zero_shot_rel_class import predict, ZSBert, load_model

from typing import List, Iterator
from spacy.tokens import Doc

class RelationsExtractorZSRC(RelationsExtractor):
    def __init__(self):
        self.model = None
        self.load_models()
        super(RelationsExtractor, self).__init__()
        
    def load_models(self,):
        if self.model is None:
            self.model = load_model()
        
    def extract_relations(self, docs: Iterator[Doc], batch_size=None):
        for doc in docs:
            items_to_process = []
            for i, e1 in enumerate(doc.ents):
                for j, e2 in enumerate(doc.ents):
                    if i == j or (e1, e2, doc.text) in items_to_process or (e2, e1, doc.text) in items_to_process:
                        continue
                    else:
                        items_to_process.append((e1, e2, doc.text))
            if len(items_to_process) > 0:
                predicted = {}
                for rel in self.relations:
                    _, probs = predict(self.model, items_to_process, rel.description, batch_size)
                    predicted[rel.name] = list(zip(probs, items_to_process))
            doc._.relations.append(predicted)


    