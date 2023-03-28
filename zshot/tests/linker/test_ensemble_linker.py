import gc
import shutil
from pathlib import Path

import pytest
import spacy

from zshot import PipelineConfig
from zshot.linker import LinkerSMXM, LinkerTARS
from zshot.linker.linker_ensemble import LinkerEnsemble
from zshot.utils.data_models import Entity


@pytest.fixture(scope="module", autouse=True)
def teardown():
    yield True
    shutil.rmtree(f"{Path.home()}/.cache/huggingface", ignore_errors=True)
    shutil.rmtree(f"{Path.home()}/.cache/zshot", ignore_errors=True)
    gc.collect()


def test_ensemble_linker_max():
    nlp = spacy.blank("en")
    nlp.add_pipe("zshot", config=PipelineConfig(
        entities=[
            Entity(name="fruits", description="The sweet and fleshy product of a tree or other plant."),
            Entity(name="fruits", description="Names of fruits such as banana, oranges")
        ],
        linker=LinkerEnsemble(
            linkers=[
                LinkerSMXM(),
                LinkerTARS(),
            ]
        )
    ), last=True)
    doc = nlp('Apple is a company name not a fruits like apples or orange')
    assert "zshot" in nlp.pipe_names
    assert len(doc.ents) > 0
    assert len(doc._.spans) > 0
    assert all([bool(ent.label_) for ent in doc.ents])
    del doc, nlp


def test_ensemble_linker_count():
    nlp = spacy.blank("en")
    nlp.add_pipe("zshot", config=PipelineConfig(
        entities=[
            Entity(name="fruits", description="The sweet and fleshy product of a tree or other plant."),
            Entity(name="fruits", description="Names of fruits such as banana, oranges")
        ],
        linker=LinkerEnsemble(
            linkers=[
                LinkerSMXM(),
                LinkerTARS(),
            ],
            strategy='count'
        )
    ), last=True)

    doc = nlp('Apple is a company name not a fruits like apples or orange')
    assert "zshot" in nlp.pipe_names
    assert len(doc.ents) > 0
    assert len(doc._.spans) > 0
    assert all([bool(ent.label_) for ent in doc.ents])
    del doc, nlp
