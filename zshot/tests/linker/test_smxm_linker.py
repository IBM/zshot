import gc
import logging
import shutil
from pathlib import Path

import pytest
import spacy

from zshot import PipelineConfig, Linker
from zshot.linker import LinkerSMXM, LinkerEnsemble
from zshot.tests.config import EX_DOCS, EX_ENTITIES
from zshot.tests.linker.test_linker import DummyLinkerEnd2End

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module", autouse=True)
def teardown():
    logger.warning("Starting smxm tests")
    yield True
    logger.warning("Removing cache")
    shutil.rmtree(f"{Path.home()}/.cache/huggingface", ignore_errors=True)
    shutil.rmtree(f"{Path.home()}/.cache/zshot", ignore_errors=True)
    gc.collect()


def test_smxm_download():
    linker = LinkerSMXM()
    linker.load_models()
    assert isinstance(linker, Linker)
    del linker.tokenizer, linker.model, linker


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
    docs = [doc for doc in nlp.pipe(EX_DOCS)]
    assert all(len(doc.ents) > 0 for doc in docs)
    del nlp.get_pipe('zshot').linker.tokenizer, nlp.get_pipe('zshot').linker.model, nlp.get_pipe('zshot').linker
    nlp.remove_pipe('zshot')
    del doc, nlp, smxm_config


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
    del nlp.get_pipe('zshot').linker.tokenizer, nlp.get_pipe('zshot').linker.model, nlp.get_pipe('zshot').linker
    nlp.remove_pipe('zshot')
    del doc, nlp, smxm_config


def test_ensemble_smxm_linker():
    nlp = spacy.blank("en")
    nlp.add_pipe("zshot", config=PipelineConfig(
        entities=EX_ENTITIES,
        linker=LinkerEnsemble(
            threshold=0.25
        )
    ), last=True)
    doc = nlp(EX_DOCS[1])
    assert len(doc.ents) > 0
    del doc, nlp
