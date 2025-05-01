import gc
import logging

import pytest
import spacy

from zshot import PipelineConfig, Linker
from zshot.linker import LinkerRelik
from zshot.tests.config import EX_DOCS, EX_ENTITIES

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module", autouse=True)
def teardown():
    logger.warning("Starting smxm tests")
    yield True
    gc.collect()


def test_relik_download():
    linker = LinkerRelik()
    linker.device = 'cpu'
    linker.load_models()
    assert isinstance(linker, Linker)
    del linker.model, linker


def test_relik_linker():
    nlp = spacy.blank("en")
    relik_config = PipelineConfig(
        linker=LinkerRelik(),
        entities=EX_ENTITIES,
        device='cpu'
    )
    nlp.add_pipe("zshot", config=relik_config, last=True)
    assert "zshot" in nlp.pipe_names

    doc = nlp(EX_DOCS[1])
    assert len(doc.ents) > 0
    docs = [doc for doc in nlp.pipe(EX_DOCS)]
    assert all(len(doc.ents) > 0 for doc in docs)
    del nlp.get_pipe('zshot').linker.model, nlp.get_pipe('zshot').linker
    nlp.remove_pipe('zshot')
    del doc, nlp, relik_config


def test_relik_linker_no_entities():
    nlp = spacy.blank("en")
    relik_config = PipelineConfig(
        linker=LinkerRelik(),
        entities=[],
        device='cpu'
    )
    nlp.add_pipe("zshot", config=relik_config, last=True)
    assert "zshot" in nlp.pipe_names

    doc = nlp(EX_DOCS[1])
    assert len(doc.ents) == 0
    del nlp.get_pipe('zshot').linker.model, nlp.get_pipe('zshot').linker
    nlp.remove_pipe('zshot')
    del doc, nlp, relik_config
