import spacy

from zshot import PipelineConfig
from zshot.linker.linker_regen.linker_regen import LinkerRegen
from zshot.mentions_extractor import MentionsExtractorSpacy
from zshot.tests.config import EX_DOCS, EX_ENTITIES


def test_regen_linker():
    nlp = spacy.load("en_core_web_sm")
    config = PipelineConfig(
        mentions_extractor=MentionsExtractorSpacy(),
        linker=LinkerRegen(),
        entities=EX_ENTITIES
    )
    nlp.add_pipe("zshot", config=config, last=True)
    assert "zshot" in nlp.pipe_names

    doc = nlp(EX_DOCS[1])
    assert len(doc.ents) > 0


def test_regen_linker_pipeline():
    nlp = spacy.load("en_core_web_sm")
    config = PipelineConfig(
        mentions_extractor=MentionsExtractorSpacy(),
        linker=LinkerRegen(),
        entities=EX_ENTITIES
    )
    nlp.add_pipe("zshot", config=config, last=True)
    assert "zshot" in nlp.pipe_names

    docs = [doc for doc in nlp.pipe(EX_DOCS)]
    assert all(len(doc.ents) > 0 for doc in docs)
