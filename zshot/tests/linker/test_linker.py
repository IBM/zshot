from typing import Iterator

import spacy
from spacy.tokens import Doc

from zshot import Linker, PipelineConfig
from zshot.utils.data_models import Span
from zshot.tests.config import EX_DOCS, EX_ENTITIES
from zshot.tests.mentions_extractor.test_mention_extractor import DummyMentionsExtractor


class DummyLinker(Linker):

    def predict(self, docs: Iterator[Doc], batch_size=None):
        return [
            [Span(mention.start_char, mention.end_char, label='label') for mention in doc._.mentions]
            for doc in docs
        ]


class DummyLinkerEnd2End(Linker):

    @property
    def is_end2end(self) -> bool:
        return True

    def predict(self, docs: Iterator[Doc], batch_size=None):
        return [[Span(0, len(doc.text) - 1, label='label')] for doc in docs]


class DummyLinkerWithEntities(Linker):

    def predict(self, docs: Iterator[Doc], batch_size=None):
        entities = self.entities
        return [
            [
                Span(mention.start_char, mention.end_char, label=entities[idx].name)
                for idx, mention in enumerate(doc._.mentions)
            ]
            for doc in docs
        ]


def test_dummy_linker():
    nlp = spacy.blank("en")
    config = PipelineConfig(
        mentions_extractor=DummyMentionsExtractor(),
        linker=DummyLinker())
    nlp.add_pipe("zshot", config=config, last=True)
    assert "zshot" in nlp.pipe_names
    doc = nlp(EX_DOCS[1])
    assert len(doc._.mentions) > 0
    assert len(doc.ents) > 0
    assert len(doc._.spans) > 0


def test_dummy_linker_with_entities_config():
    nlp = spacy.blank("en")

    nlp.add_pipe("zshot", config=PipelineConfig(
        mentions_extractor=DummyMentionsExtractor(),
        linker=DummyLinker(),
        entities=EX_ENTITIES), last=True)

    assert "zshot" in nlp.pipe_names
    doc = nlp(EX_DOCS[1])

    assert len(doc._.mentions) > 0
    assert len(doc.ents) > 0
    assert len(doc._.spans) > 0
    assert all([bool(ent.label_) for ent in doc.ents])


def test_dummy_linker_end2end():
    nlp = spacy.blank("en")

    nlp.add_pipe("zshot", config=PipelineConfig(
        mentions_extractor=DummyMentionsExtractor(),
        linker=DummyLinkerEnd2End(),
        entities=EX_ENTITIES), last=True)

    assert "zshot" in nlp.pipe_names
    doc = nlp(EX_DOCS[1])

    assert len(doc._.mentions) == 0
    assert len(doc.ents) > 0
    assert len(doc._.spans) > 0
