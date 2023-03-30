import gc
import pkgutil

import pytest
import spacy

from zshot import PipelineConfig, Linker
from zshot.linker import LinkerTARS
from zshot.tests.config import EX_DOCS, EX_ENTITIES
from zshot.utils.data_models import Entity

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
    linker = LinkerTARS()
    linker.load_models()
    assert isinstance(linker, Linker)
    del linker.model, linker


def test_tars_end2end_with_entities():
    if not pkgutil.find_loader("flair"):
        return
    nlp = spacy.blank("en")

    config_zshot = PipelineConfig(linker=LinkerTARS(), entities=EX_ENTITIES)
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    doc = nlp(EX_DOCS[1])
    assert doc.ents != ()
    del nlp.get_pipe('zshot').linker.model, nlp.get_pipe('zshot').linker
    nlp.remove_pipe('zshot')
    del nlp, config_zshot


def test_tars_end2end_pipeline_with_entities():
    if not pkgutil.find_loader("flair"):
        return
    nlp = spacy.blank("en")

    config_zshot = PipelineConfig(linker=LinkerTARS(), entities=EX_ENTITIES)
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    docs = [doc for doc in nlp.pipe(EX_DOCS)]
    assert all(doc.ents != () for doc in docs)
    del nlp.get_pipe('zshot').linker.model, nlp.get_pipe('zshot').linker
    nlp.remove_pipe('zshot')
    del nlp, config_zshot


def test_tars_end2end_overlap():
    if not pkgutil.find_loader("flair"):
        return
    nlp = spacy.blank("en")

    config_zshot = PipelineConfig(linker=LinkerTARS(), entities=["company", "location", "organic compound"])
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    doc = nlp(OVERLAP_TEXT)
    assert len(doc.ents) > 0
    del nlp.get_pipe('zshot').linker.model, nlp.get_pipe('zshot').linker
    nlp.remove_pipe('zshot')
    del nlp, config_zshot


def test_tars_end2end_incomplete_spans():
    if not pkgutil.find_loader("flair"):
        return
    nlp = spacy.blank("en")

    config_zshot = PipelineConfig(linker=LinkerTARS())
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    doc = nlp(INCOMPLETE_SPANS_TEXT)
    assert len(doc.ents) > 0
    del nlp.get_pipe('zshot').linker.model, nlp.get_pipe('zshot').linker
    nlp.remove_pipe('zshot')
    del nlp, config_zshot


def test_flat_entities():
    linker_tars = LinkerTARS()

    # Dict
    entities = {
        "company": "Company entity description",
        "location": "Location entity description",
        "organic compound": "Organic compound entity description"
    }
    linker_tars.set_kg(entities)
    assert linker_tars.entities == ["company", "location", "organic compound"]

    # List of strings
    entities = ["company", "location", "organic compound"]
    linker_tars.set_kg(entities)
    assert linker_tars.entities == ["company", "location", "organic compound"]

    # List of entities
    entities = [
        Entity(name="company", description="Company entity description"),
        Entity(name="location", description="Location entity description"),
        Entity(name="organic compound", description="Organic compound entity description")
    ]
    linker_tars.set_kg(entities)
    assert linker_tars.entities == ["company", "location", "organic compound"]

    # None
    linker_tars.set_kg(None)
    assert linker_tars.entities == []
