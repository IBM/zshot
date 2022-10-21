import gc
import logging
import shutil
from pathlib import Path

import pytest
import spacy

from zshot import PipelineConfig, MentionsExtractor
from zshot.mentions_extractor import MentionsExtractorSMXM
from zshot.tests.config import EX_DOCS, EX_ENTITIES

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
    mentions_extractor = MentionsExtractorSMXM()
    mentions_extractor.load_models()
    assert isinstance(mentions_extractor, MentionsExtractor)
    del mentions_extractor


def test_smxm_mentions_extractor():
    nlp = spacy.blank("en")
    smxm_config = PipelineConfig(
        mentions_extractor=MentionsExtractorSMXM(),
        mentions=EX_ENTITIES
    )
    nlp.add_pipe("zshot", config=smxm_config, last=True)
    assert "zshot" in nlp.pipe_names

    doc = nlp(EX_DOCS[1])
    assert len(doc._.mentions) > 0
    nlp.remove_pipe('zshot')
    del doc, nlp


def test_smxm_mentions_extractor_pipeline():
    nlp = spacy.blank("en")
    smxm_config = PipelineConfig(
        mentions_extractor=MentionsExtractorSMXM(),
        mentions=EX_ENTITIES
    )
    nlp.add_pipe("zshot", config=smxm_config, last=True)
    assert "zshot" in nlp.pipe_names

    docs = [doc for doc in nlp.pipe(EX_DOCS)]
    assert all(len(doc._.mentions) > 0 for doc in docs)
    nlp.remove_pipe('zshot')
    del docs, nlp


def test_smxm_mentions_extractor_no_entities():
    nlp = spacy.blank("en")
    smxm_config = PipelineConfig(
        mentions_extractor=MentionsExtractorSMXM(),
        mentions=[]
    )
    nlp.add_pipe("zshot", config=smxm_config, last=True)
    assert "zshot" in nlp.pipe_names

    doc = nlp(EX_DOCS[1])
    assert len(doc._.mentions) == 0
    nlp.remove_pipe('zshot')
    del doc, nlp
