from typing import Iterator, List

import spacy
from spacy.tokens import Doc

from zshot import Linker
from zshot.entity import Entity
from zshot.tests.config import EX_DOCS, EX_ENTITIES
from zshot.tests.mentions_extractor.test_mention_extractor import DummyMentionsExtractor


class DummyLinker(Linker):

    def link(self, docs: Iterator[Doc], batch_size=None):
        for doc in docs:
            for mention in doc._.mentions:
                doc.ents += (doc.char_span(mention.start_char, mention.end_char, label='label'),)


class DummyLinkerEnd2End(Linker):

    @property
    def is_end2end(self) -> bool:
        return True

    def link(self, docs: Iterator[Doc], batch_size=None):
        for doc in docs:
            doc.ents += (doc.char_span(0, len(doc.text)-1, label='label'),)


class DummyLinkerWithEntities(Linker):

    def link(self, docs: Iterator[Doc], batch_size=None):
        entities = self.entities
        for doc in docs:
            for idx, mention in enumerate(doc._.mentions):
                doc.ents += (doc.char_span(mention.start_char, mention.end_char,
                                           label=entities[idx % len(entities)].name),)


def test_dummy_linker():
    @spacy.registry.misc("dummy.mentions-extractor")
    def create_custom_spacy_extractor():
        return DummyMentionsExtractor()

    @spacy.registry.misc("dummy.linker")
    def create_custom_linker():
        return DummyLinker()

    nlp = spacy.load("en_core_web_sm")
    config_zshot = {
        "mentions_extractor": {"@misc": "dummy.mentions-extractor"},
        "linker": {"@misc": "dummy.linker"}
    }
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    doc = nlp(EX_DOCS[1])
    assert len(doc._.mentions) > 0
    assert len(doc.ents) > 0


def test_dummy_linker_with_entities_config():
    @spacy.registry.misc("dummy.mentions-extractor")
    def create_custom_spacy_extractor():
        return DummyMentionsExtractor()

    @spacy.registry.misc("dummy.linker")
    def create_custom_linker():
        return DummyLinker()

    @spacy.registry.misc("get.entities.v1")
    def get_entities() -> List[Entity]:
        return EX_ENTITIES

    nlp = spacy.load("en_core_web_sm")
    config_zshot = {
        "mentions_extractor": {"@misc": "dummy.mentions-extractor"},
        "linker": {"@misc": "dummy.linker"},
        "entities": {"@misc": "get.entities.v1"}
    }

    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    doc = nlp(EX_DOCS[1])

    assert len(doc._.mentions) > 0
    assert len(doc.ents) > 0
    assert all([bool(ent.label_) for ent in doc.ents])


def test_dummy_linker_end2end():
    @spacy.registry.misc("dummy.mentions-extractor")
    def create_custom_spacy_extractor():
        return DummyMentionsExtractor()

    @spacy.registry.misc("dummy.linker")
    def create_custom_linker():
        return DummyLinkerEnd2End()

    @spacy.registry.misc("get.entities.v1")
    def get_entities() -> List[Entity]:
        return EX_ENTITIES

    nlp = spacy.load("en_core_web_sm")
    config_zshot = {
        "mentions_extractor": {"@misc": "dummy.mentions-extractor"},
        "linker": {"@misc": "dummy.linker"},
        "entities": {"@misc": "get.entities.v1"}
    }

    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    doc = nlp(EX_DOCS[1])

    assert len(doc._.mentions) == 0
    assert len(doc.ents) > 0
