import spacy

from zshot import PipelineConfig
from zshot.linker.linker_ensemble import LinkerEnsemble
from zshot.tests.linker.test_linker import DummyLinkerEnd2End
from zshot.utils.data_models import Entity


def test_ensemble_linker_max():
    nlp = spacy.blank("en")
    nlp.add_pipe("zshot", config=PipelineConfig(
        entities=[
            Entity(name="fruits", description="The sweet and fleshy product of a tree or other plant."),
            Entity(name="fruits", description="Names of fruits such as banana, oranges")
        ],
        linker=LinkerEnsemble(
            linkers=[
                DummyLinkerEnd2End(),
                DummyLinkerEnd2End(),
            ]
        )
    ), last=True)
    doc = nlp('Apple is a company name not a fruits like apples or orange')
    assert "zshot" in nlp.pipe_names
    assert len(doc.ents) > 0
    assert len(doc._.spans) > 0
    assert all([bool(ent.label_) for ent in doc.ents])


def test_ensemble_linker_count():
    nlp = spacy.blank("en")
    nlp.add_pipe("zshot", config=PipelineConfig(
        entities=[
            Entity(name="fruits", description="The sweet and fleshy product of a tree or other plant."),
            Entity(name="fruits", description="Names of fruits such as banana, oranges")
        ],
        linker=LinkerEnsemble(
            linkers=[
                DummyLinkerEnd2End(),
                DummyLinkerEnd2End(),
            ],
            strategy='count'
        )
    ), last=True)

    doc = nlp('Apple is a company name not a fruits like apples or orange')
    assert "zshot" in nlp.pipe_names
    assert len(doc.ents) > 0
    assert len(doc._.spans) > 0
    assert all([bool(ent.label_) for ent in doc.ents])
