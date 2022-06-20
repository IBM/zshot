from typing import Iterator

import spacy
from spacy.tokens.doc import Doc

from zshot import MentionsExtractor, PipelineConfig
from zshot.tests.config import EX_DOCS, EX_ENTITIES


class DummyMentionsExtractor(MentionsExtractor):
    def extract_mentions(self, docs: Iterator[Doc], batch_size=None):
        for doc in docs:
            doc._.mentions.append(doc.char_span(0, len(doc.text) - 1))


class DummyMentionsExtractorWithEntities(MentionsExtractor):
    def extract_mentions(self, docs: Iterator[Doc], batch_size=None):
        for idx, doc in enumerate(docs):
            doc._.mentions.append(doc.char_span(0, len(doc.text) - 1, label=self.entities[idx % len(self.entities)].name))


def test_dummy_mentions_extractor():
    nlp = spacy.blank("en")
    config_zshot = PipelineConfig(mentions_extractor=DummyMentionsExtractor())
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    doc = nlp(EX_DOCS[1])
    assert doc.ents == ()
    assert len(doc._.mentions) > 0


def test_dummy_mentions_extractor_with_entities_config():
    nlp = spacy.blank("en")
    config_zshot = PipelineConfig(mentions_extractor=DummyMentionsExtractorWithEntities(),
                                  entities=EX_ENTITIES)
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    doc = nlp(EX_DOCS[1])
    assert doc.ents == ()
    assert len(doc._.mentions) > 0
    assert all([bool(m.label_) for m in doc._.mentions])
