import gc
import logging
import shutil
from pathlib import Path

import pytest
import spacy

from zshot import PipelineConfig
from zshot.linker.linker_regen.linker_regen import LinkerRegen
from zshot.mentions_extractor import MentionsExtractorSpacy
from zshot.tests.config import EX_DOCS, EX_ENTITIES

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module", autouse=True)
def teardown():
    logger.warning("Starting regen tests")
    yield True
    logger.warning("Removing cache")
    shutil.rmtree(f"{Path.home()}/.cache/huggingface", ignore_errors=True)
    shutil.rmtree(f"{Path.home()}/.cache/zshot", ignore_errors=True)
    gc.collect()


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
    del nlp.get_pipe('zshot').mentions_extractor, nlp.get_pipe('zshot').entities, nlp.get_pipe('zshot').nlp
    del nlp.get_pipe('zshot').linker.tokenizer, nlp.get_pipe('zshot').linker.trie, \
        nlp.get_pipe('zshot').linker.model, nlp.get_pipe('zshot').linker
    nlp.remove_pipe('zshot')
    del doc, nlp, config


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
    del nlp.get_pipe('zshot').mentions_extractor, nlp.get_pipe('zshot').entities, nlp.get_pipe('zshot').nlp
    del nlp.get_pipe('zshot').linker.tokenizer, nlp.get_pipe('zshot').linker.trie, \
        nlp.get_pipe('zshot').linker.model, nlp.get_pipe('zshot').linker
    nlp.remove_pipe('zshot')
    del docs, nlp, config
