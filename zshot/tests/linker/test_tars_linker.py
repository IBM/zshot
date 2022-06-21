import pkgutil

import spacy

from zshot.linker import TARSLinker
from zshot.tests.config import EX_DOCS, EX_ENTITIES
from zshot import PipelineConfig


def test_tars_end2end_no_entities():
    if not pkgutil.find_loader("flair"):
        return
    nlp = spacy.load("en_core_web_sm")

    config_zshot = PipelineConfig(linker=TARSLinker())
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    doc = nlp(EX_DOCS[1])
    assert doc.ents == ()


def test_tars_end2end_pipeline_no_entities():
    if not pkgutil.find_loader("flair"):
        return
    nlp = spacy.load("en_core_web_sm")

    config_zshot = PipelineConfig(linker=TARSLinker())
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    docs = [doc for doc in nlp.pipe(EX_DOCS)]
    assert all(doc.ents == () for doc in docs)


def test_tars_end2end_with_entities():
    if not pkgutil.find_loader("flair"):
        return
    nlp = spacy.load("en_core_web_sm")

    config_zshot = PipelineConfig(linker=TARSLinker(), entities=EX_ENTITIES)
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    doc = nlp(EX_DOCS[1])
    assert doc.ents != ()


def test_tars_end2end_pipeline_with_entities():
    if not pkgutil.find_loader("flair"):
        return
    nlp = spacy.load("en_core_web_sm")

    config_zshot = PipelineConfig(linker=TARSLinker(), entities=EX_ENTITIES)
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    docs = [doc for doc in nlp.pipe(EX_DOCS)]
    assert all(doc.ents != () for doc in docs)
