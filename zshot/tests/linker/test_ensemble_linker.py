import spacy

from zshot import PipelineConfig
from zshot.linker import LinkerSMXM, LinkerTARS
from zshot.tests.config import EX_DOCS, EX_ENTITIES
from zshot.linker.linker_ensemble import LinkerEnsemble
from zshot.utils.data_models import Entity


def test_ensemble_linker_with_entities_config():
    nlp = spacy.blank("en")
    nlp.add_pipe("zshot", config=PipelineConfig(
        linker=LinkerEnsemble(
            enhance_entities=[
                [Entity(name="fruits", description="The sweet and fleshy product of a tree or other plant.")],
                [Entity(name="fruits", description="Names of fruits such as banana, oranges")]
            ],
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