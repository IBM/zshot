from typing import Iterator

import spacy
from spacy.tokens import Doc

from zshot import PipelineConfig, Linker
from zshot.relation_extractor import RelationsExtractorZSRC
from zshot.tests.config import EX_RELATIONS, EX_DATASET_RELATIONS, EX_DOCS
from zshot.utils.data_models import Span


class DummyLinkerEnd2End(Linker):
    def predict(self, docs: Iterator[Doc], batch_size=None):
        return [[Span(187, 165, label='label', score=0.9),
                 Span(111, 129, label='label', score=0.9)] for doc in docs]


def test_zsrc_with_entities_config_dummy_annotator():
    nlp = spacy.blank("en")
    config_zshot = PipelineConfig(
        linker=DummyLinkerEnd2End(),
        relations_extractor=RelationsExtractorZSRC(),
        relations=EX_RELATIONS,
    )
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    doc = nlp(EX_DOCS[1])
    assert len(doc._.relations) == 0
    doc = nlp(EX_DATASET_RELATIONS['sentences'][0])
    assert len(doc._.relations) == 1
