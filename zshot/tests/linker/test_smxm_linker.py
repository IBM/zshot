import logging
import shutil
from pathlib import Path

import pytest
import spacy

from zshot import PipelineConfig, Linker
from zshot.linker import LinkerSMXM
from zshot.tests.config import EX_DOCS, EX_ENTITIES

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module", autouse=True)
def teardown():
    logger.warning("Starting smxm tests")
    yield True
    logger.warning("Removing cache")
    shutil.rmtree(f"{Path.home()}/.cache/huggingface", ignore_errors=True)
    shutil.rmtree(f"{Path.home()}/.cache/zshot", ignore_errors=True)


def test_smxm_download():
    linker = LinkerSMXM()
    linker.load_models()
    assert isinstance(linker, Linker)


def test_smxm_linker():
    nlp = spacy.blank("en")
    smxm_config = PipelineConfig(
        linker=LinkerSMXM(),
        entities=EX_ENTITIES
    )
    nlp.add_pipe("zshot", config=smxm_config, last=True)
    assert "zshot" in nlp.pipe_names

    doc = nlp(EX_DOCS[1])
    assert len(doc.ents) > 0


def test_smxm_linker_pipeline():
    nlp = spacy.blank("en")
    smxm_config = PipelineConfig(
        linker=LinkerSMXM(),
        entities=EX_ENTITIES
    )
    nlp.add_pipe("zshot", config=smxm_config, last=True)
    assert "zshot" in nlp.pipe_names

    docs = [doc for doc in nlp.pipe(EX_DOCS)]
    assert all(len(doc.ents) > 0 for doc in docs)


def test_smxm_linker_no_entities():
    nlp = spacy.blank("en")
    smxm_config = PipelineConfig(
        linker=LinkerSMXM(),
        entities=[]
    )
    nlp.add_pipe("zshot", config=smxm_config, last=True)
    assert "zshot" in nlp.pipe_names

    doc = nlp(EX_DOCS[1])
    assert len(doc.ents) == 0
