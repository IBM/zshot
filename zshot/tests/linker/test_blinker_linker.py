import gc
import logging
import pkgutil

import pytest
import spacy

from zshot import PipelineConfig, Linker
from zshot.linker import LinkerBlink
from zshot.tests.config import EX_DOCS, EX_ENTITIES

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module", autouse=True)
def teardown():
    logger.warning("Starting blink tests")
    yield True
    gc.collect()


@pytest.mark.skipif(not pkgutil.find_loader("blink"), reason="BLINK is not installed")
def test_blink():
    linker = LinkerBlink()
    with pytest.raises(Exception):
        assert len(linker.entities_list) > 1
    with pytest.raises(Exception):
        assert len(linker.local_id2wikipedia_id) > 1
    with pytest.raises(Exception):
        assert linker.local_name2wikipedia_url('IBM').startswith("https://en.wikipedia.org/wiki")


@pytest.mark.skip(reason="Too expensive to run on every commit")
def test_blink_download():
    linker = LinkerBlink()
    linker.load_models()
    assert isinstance(linker, Linker)
    del linker.tokenizer, linker.model, linker


@pytest.mark.skip(reason="Too expensive to run on every commit")
def test_blink_linker():
    nlp = spacy.blank("en")
    blink_config = PipelineConfig(
        linker=LinkerBlink(),
        entities=EX_ENTITIES
    )
    nlp.add_pipe("zshot", config=blink_config, last=True)
    assert "zshot" in nlp.pipe_names

    doc = nlp(EX_DOCS[1])
    assert len(doc.ents) > 0
    docs = [doc for doc in nlp.pipe(EX_DOCS)]
    assert all(len(doc.ents) > 0 for doc in docs)
    del nlp.get_pipe('zshot').linker.tokenizer, nlp.get_pipe('zshot').linker.model, nlp.get_pipe('zshot').linker
    nlp.remove_pipe('zshot')
    del doc, nlp, blink_config


@pytest.mark.skip(reason="Too expensive to run on every commit")
def test_blink_linker_no_entities():
    nlp = spacy.blank("en")
    blink_config = PipelineConfig(
        linker=LinkerBlink(),
        entities=[]
    )
    nlp.add_pipe("zshot", config=blink_config, last=True)
    assert "zshot" in nlp.pipe_names

    doc = nlp(EX_DOCS[1])
    assert len(doc.ents) == 0
    del nlp.get_pipe('zshot').linker.tokenizer, nlp.get_pipe('zshot').linker.model, nlp.get_pipe('zshot').linker
    nlp.remove_pipe('zshot')
    del doc, nlp, blink_config
