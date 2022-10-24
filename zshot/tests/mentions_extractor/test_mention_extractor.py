from typing import Iterator

import spacy
from spacy.tokens.doc import Doc

from zshot import MentionsExtractor, PipelineConfig
from zshot.tests.config import EX_DOCS, EX_ENTITIES
from zshot.utils.data_models import Span


class DummyMentionsExtractor(MentionsExtractor):
    def predict(self, docs: Iterator[Doc], batch_size=None):
        return [
            [Span(0, len(doc.text) - 1)]
            for doc in docs
        ]


class DummyMentionsExtractorWithNER(MentionsExtractor):
    @property
    def require_existing_ner(self) -> bool:
        return True

    def predict(self, docs: Iterator[Doc], batch_size=None):
        return [
            [Span(0, len(doc.text) - 1)]
            for doc in docs
        ]


class DummyMentionsExtractorWithEntities(MentionsExtractor):
    def predict(self, docs: Iterator[Doc], batch_size=None):
        return [
            [Span(0, len(doc.text) - 1)]
            for idx, doc in enumerate(docs)
        ]


def test_dummy_mentions_extractor():
    nlp = spacy.blank("en")
    config_zshot = PipelineConfig(mentions_extractor=DummyMentionsExtractorWithEntities())
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    doc = nlp(EX_DOCS[1])
    assert doc.ents == ()
    assert len(doc._.mentions) > 0
    del doc, nlp


def test_dummy_mentions_extractor_with_entities_config():
    nlp = spacy.blank("en")
    config_zshot = PipelineConfig(mentions_extractor=DummyMentionsExtractorWithEntities(),
                                  mentions=EX_ENTITIES)
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    doc = nlp(EX_DOCS[1])
    assert doc.ents == ()
    assert len(doc._.mentions) > 0
    del doc, nlp
