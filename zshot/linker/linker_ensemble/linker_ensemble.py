from typing import Iterator, Optional, List

from spacy.tokens import Doc

from zshot.linker import Linker
from zshot.linker import LinkerSMXM
from zshot.utils.ensembler import Ensembler
from zshot.utils.data_models import Entity
from zshot.linker.linker_ensemble.utils import sub_span_scoring_per_description, get_enhance_entities


class LinkerEnsemble(Linker):
    def __init__(self,
                 linkers: Optional[List[Linker]] = None,
                 strategy: Optional[str] = 'max',
                 threshold: Optional[float] = 0.5):
        """ Ensemble of linkers and entities to improve performance.
            Each combination of linker with entity will be a voter.

        :param linkers: Linkers to use in the ensemble
        :param strategy: Strategy to use. Options: max; count
            When `max` choose the label with max total vote score
            When `count` choose the label with max total vote count
        :param threshold: Threshold to use. Proportion of voters voting the entity
        """
        super(LinkerEnsemble, self).__init__()
        if linkers is not None:
            self.linkers = linkers
        else:
            # default options
            self.linkers = [
                LinkerSMXM()
            ]
        self.enhance_entities = []
        self.strategy = strategy
        self.threshold = threshold
        self.ensembler = None

    def set_smxm_model(self, smxm_model):
        for linker in self.linkers:
            if isinstance(linker, LinkerSMXM):
                linker.model_name = smxm_model

    def set_kg(self, entities: Iterator[Entity]):
        """
        Set entities that linker can use
        :param entities: The list of entities
        """
        super().set_kg(entities)
        self.enhance_entities = get_enhance_entities(self.entities)
        self.ensembler = Ensembler(len(self.linkers),
                                   len(self.enhance_entities) if self.enhance_entities is not None else -1,
                                   threshold=self.threshold)
        for linker in self.linkers:
            linker.set_kg(entities)

    def predict(self, docs: Iterator[Doc], batch_size=None):
        """
        Perform the entity prediction
        :param docs: A list of spacy Document
        :param batch_size: The batch size
        :return: List Spans for each Document in docs
        """
        spans = []
        if self.enhance_entities is not None:
            for entities in self.enhance_entities:
                self.set_kg(entities)
                for linker in self.linkers:
                    span_prediction = linker.predict(docs, batch_size)
                    spans.append(span_prediction)
        else:
            for linker in self.linkers:
                span_prediction = linker.predict(docs, batch_size)
                spans.append(span_prediction)
        return self.prediction_ensemble(spans)

    def prediction_ensemble(self, spans):
        doc_ensemble_spans = []
        num_doc = len(spans[0])
        for doc_idx in range(num_doc):
            union_spans = {}
            span_per_descriptions = []
            for span in spans:
                span_per_descriptions.append(span[doc_idx])
                for s in span[doc_idx]:
                    span_pos = (s.start, s.end)
                    if span_pos not in union_spans:
                        union_spans[span_pos] = [s]
                    else:
                        union_spans[span_pos].append(s)

            sub_span_scoring_per_description(union_spans, span_per_descriptions)
            all_union_spans = self.ensembler.ensemble(union_spans)
            doc_ensemble_spans.append(all_union_spans)

        return doc_ensemble_spans
