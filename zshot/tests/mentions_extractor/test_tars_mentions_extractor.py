import gc
import pkgutil

import pytest
import spacy

from zshot import PipelineConfig, MentionsExtractor
from zshot.mentions_extractor import MentionsExtractorTARS
from zshot.tests.config import EX_DOCS, EX_ENTITIES

OVERLAP_TEXT = "Senator McConnell in addition to this the New York Times editors wrote in reaction to the " \
               "Supreme Court 's decision striking down the military tribunal set up to private detainees " \
               "being held in Guantanamo bay it is far more than a narrow ruling on the issue of military courts . " \
               "It is an important and welcome reaffirmation That even in times of war the law is what the " \
               "constitution the statuette books and the Geneva convention say it is . " \
               "Not what the President wants it to be /."
INCOMPLETE_SPANS_TEXT = "-LSB- -LSB- They attacked small bridges and small districts , " \
                        "and generally looted these stations . -RSB- -RSB-"


@pytest.fixture(scope="module", autouse=True)
def teardown():
    yield True
    gc.collect()


def test_tars_download():
    mentions_extractor = MentionsExtractorTARS()
    mentions_extractor.load_models()
    assert isinstance(mentions_extractor, MentionsExtractor)
    del mentions_extractor


def test_tars_mentions_extractor_with_entities():
    if not pkgutil.find_loader("flair"):
        return
    nlp = spacy.blank("en")

    config_zshot = PipelineConfig(mentions_extractor=MentionsExtractorTARS(), mentions=EX_ENTITIES)
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    doc = nlp(EX_DOCS[1])
    assert doc._.mentions != ()
    nlp.remove_pipe('zshot')
    del doc, nlp


def test_tars_mentions_extractor_pipeline_with_entities():
    if not pkgutil.find_loader("flair"):
        return
    nlp = spacy.blank("en")

    config_zshot = PipelineConfig(mentions_extractor=MentionsExtractorTARS(), mentions=EX_ENTITIES)
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    docs = [doc for doc in nlp.pipe(EX_DOCS)]
    assert all(doc._.mentions != () for doc in docs)
    nlp.remove_pipe('zshot')
    del docs, nlp


def test_tars_mentions_extractor_overlap():
    if not pkgutil.find_loader("flair"):
        return
    nlp = spacy.blank("en")

    config_zshot = PipelineConfig(mentions_extractor=MentionsExtractorTARS(),
                                  mentions=["company", "location", "organic compound"])
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    doc = nlp(OVERLAP_TEXT)
    assert len(doc._.mentions) > 0
    nlp.remove_pipe('zshot')
    del doc, nlp


def test_tars_end2end_incomplete_spans():
    if not pkgutil.find_loader("flair"):
        return
    nlp = spacy.blank("en")

    config_zshot = PipelineConfig(mentions_extractor=MentionsExtractorTARS())
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    doc = nlp(INCOMPLETE_SPANS_TEXT)
    assert len(doc._.mentions) > 0
    nlp.remove_pipe('zshot')
    del doc, nlp
