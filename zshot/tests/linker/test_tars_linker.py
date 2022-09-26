import pkgutil

import spacy

from zshot.linker import LinkerTARS
from zshot.tests.config import EX_DOCS, EX_ENTITIES
from zshot import PipelineConfig

OVERLAP_TEXT = "Senator McConnell in addition to this the New York Times editors wrote in reaction to the " \
               "Supreme Court 's decision striking down the military tribunal set up to private detainees " \
               "being held in Guantanamo bay it is far more than a narrow ruling on the issue of military courts . " \
               "It is an important and welcome reaffirmation That even in times of war the law is what the " \
               "constitution the statuette books and the Geneva convention say it is . " \
               "Not what the President wants it to be /."
INCOMPLETE_SPANS_TEXT = "-LSB- -LSB- They attacked small bridges and small districts , " \
                        "and generally looted these stations . -RSB- -RSB-"


def test_tars_end2end_with_entities():
    if not pkgutil.find_loader("flair"):
        return
    nlp = spacy.blank("en")

    config_zshot = PipelineConfig(linker=LinkerTARS(), entities=EX_ENTITIES)
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    doc = nlp(EX_DOCS[1])
    assert doc.ents != ()


def test_tars_end2end_pipeline_with_entities():
    if not pkgutil.find_loader("flair"):
        return
    nlp = spacy.blank("en")

    config_zshot = PipelineConfig(linker=LinkerTARS(), entities=EX_ENTITIES)
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    docs = [doc for doc in nlp.pipe(EX_DOCS)]
    assert all(doc.ents != () for doc in docs)


def test_tars_end2end_overlap():
    if not pkgutil.find_loader("flair"):
        return
    nlp = spacy.blank("en")

    config_zshot = PipelineConfig(linker=LinkerTARS(), entities=["company", "location", "organic compound"])
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    nlp(OVERLAP_TEXT)


def test_tars_end2end_incomplete_spans():
    if not pkgutil.find_loader("flair"):
        return
    nlp = spacy.blank("en")

    config_zshot = PipelineConfig(linker=LinkerTARS())
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    doc = nlp(INCOMPLETE_SPANS_TEXT)
    assert len(doc.ents) > 0
