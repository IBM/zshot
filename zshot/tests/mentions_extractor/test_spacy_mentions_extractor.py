import spacy

from zshot.mentions_extractor import MentionsExtractorSpacy
from zshot.mentions_extractor.mentions_extractor_spacy import ExtractorType
from zshot.tests.config import EX_DOCS
from zshot import PipelineConfig


def test_spacy_ner_mentions_extractor():
    nlp = spacy.load("en_core_web_sm")

    config_zshot = PipelineConfig(mentions_extractor=MentionsExtractorSpacy(ExtractorType.NER))
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names and "ner" in nlp.pipe_names

    doc = nlp(EX_DOCS[1])
    assert doc.ents == ()
    assert len(doc._.mentions) > 0
    nlp = None


def test_custom_spacy_mentions_extractor():
    nlp = spacy.load("en_core_web_sm")

    custom_component = MentionsExtractorSpacy(ExtractorType.NER)
    config_zshot = PipelineConfig(mentions_extractor=custom_component, disable_default_ner=False)
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names and "ner" in nlp.pipe_names

    doc = nlp(EX_DOCS[1])
    assert doc.ents == ()
    assert len(doc._.mentions) > 0
    nlp = None


def test_spacy_pos_mentions_extractor():
    nlp = spacy.load("en_core_web_sm")

    config_zshot = PipelineConfig(mentions_extractor=MentionsExtractorSpacy(ExtractorType.POS))
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names and "ner" not in nlp.pipe_names
    doc = nlp(EX_DOCS[1])
    assert doc.ents == ()
    assert len(doc._.mentions) > 0
    nlp = None


def test_spacy_ner_mentions_extractor_pipeline():
    nlp = spacy.load("en_core_web_sm")

    config_zshot = PipelineConfig(mentions_extractor=MentionsExtractorSpacy(ExtractorType.NER))
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names and "ner" in nlp.pipe_names
    docs = [doc for doc in nlp.pipe(EX_DOCS)]
    assert all(doc.ents == () for doc in docs)
    assert all(len(doc._.mentions) > 0 for doc in docs)
    nlp = None


def test_spacy_pos_mentions_extractor_pipeline():
    nlp = spacy.load("en_core_web_sm")

    config_zshot = PipelineConfig(mentions_extractor=MentionsExtractorSpacy(ExtractorType.POS))
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names and "ner" not in nlp.pipe_names
    docs = [doc for doc in nlp.pipe(EX_DOCS)]
    assert all(doc.ents == () for doc in docs)
    assert all(len(doc._.mentions) > 0 for doc in docs)
    nlp = None
