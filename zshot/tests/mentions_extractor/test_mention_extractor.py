from typing import List

import spacy
from spacy.tokens.doc import Doc

from zshot import MentionsExtractor
from zshot.entity import Entity
from zshot.tests.config import EX_DOCS, EX_ENTITIES


class DummyMentionsExtractor(MentionsExtractor):
    def extract_mentions(self, docs: List[Doc], batch_size=None):
        for doc in docs:
            doc._.mentions.append(doc.char_span(0, len(doc.text)-1))


class DummyMentionsExtractorWithEntities(MentionsExtractor):
    def extract_mentions(self, docs: List[Doc], batch_size=None):
        for idx, doc in enumerate(docs):
            doc._.mentions.append(doc.char_span(0, len(doc.text)-1, label=self.entities[idx % len(self.entities)].name))


def test_dummy_mentions_extractor():
    @spacy.registry.misc("dummy.mentions-extractor")
    def create_custom_spacy_extractor():
        return DummyMentionsExtractor()

    nlp = spacy.load("en_core_web_sm")
    config_zshot = {
        "mentions_extractor": {"@misc": "dummy.mentions-extractor"}
    }
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    doc = nlp(EX_DOCS[1])
    assert doc.ents == ()
    assert len(doc._.mentions) > 0


def test_dummy_mentions_extractor_with_entities_config():
    @spacy.registry.misc("dummy.mentions-extractor")
    def create_custom_spacy_extractor():
        return DummyMentionsExtractorWithEntities()

    @spacy.registry.misc("get.entities.v1")
    def get_entities() -> List[Entity]:
        return EX_ENTITIES

    nlp = spacy.load("en_core_web_sm")
    config_zshot = {
        "mentions_extractor": {"@misc": "dummy.mentions-extractor"},
        "entities": {"@misc": "get.entities.v1"}
    }

    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    doc = nlp(EX_DOCS[1])
    assert doc.ents == ()
    assert len(doc._.mentions) > 0
    assert all([bool(m.label_) for m in doc._.mentions])
