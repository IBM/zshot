import pytest
import spacy

from zshot import PipelineConfig
from zshot.linker import LinkerSMXM
from zshot.utils.data_models import Entity
from zshot.utils.enrichment.description_enrichment import PreTrainedLMExtensionStrategy, \
    FineTunedLMExtensionStrategy, SummarizationStrategy, ParaphrasingStrategy, EntropyHeuristic


@pytest.mark.skip(reason="Too expensive to run on every commit")
def test_pretrained_lm_extension_strategy():
    description = "The name of a company"
    strategy = PreTrainedLMExtensionStrategy()
    num_variations = 3

    desc_variations = strategy.alter_description(
        description, num_variations=num_variations
    )

    assert len(desc_variations) == 3 and len(set(desc_variations + [description])) == 4


@pytest.mark.skip(reason="Too expensive to run on every commit")
def test_finetuned_lm_extension_strategy():
    description = "The name of a company"
    strategy = FineTunedLMExtensionStrategy()
    num_variations = 3

    desc_variations = strategy.alter_description(
        description, num_variations=num_variations
    )

    assert len(desc_variations) == 3 and len(set(desc_variations + [description])) == 4


@pytest.mark.skip(reason="Too expensive to run on every commit")
def test_summarization_strategy():
    description = "The name of a company"
    strategy = SummarizationStrategy()
    num_variations = 3

    desc_variations = strategy.alter_description(
        description, num_variations=num_variations
    )

    assert len(desc_variations) == 3 and len(set(desc_variations + [description])) == 4


@pytest.mark.skip(reason="Too expensive to run on every commit")
def test_paraphrasing_strategy():
    description = "The name of a company"
    strategy = ParaphrasingStrategy()
    num_variations = 3

    desc_variations = strategy.alter_description(
        description, num_variations=num_variations
    )

    assert len(desc_variations) == 3 and len(set(desc_variations + [description])) == 4


@pytest.mark.skip(reason="Too expensive to run on every commit")
def test_entropy_heuristic():
    def check_is_tuple(x):
        return isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str) and isinstance(x[1], float)

    entropy_heuristic = EntropyHeuristic()
    dataset = [
        {'tokens': ['IBM', 'headquarters', 'are', 'located', 'in', 'Armonk', '.'],
         'ner_tags': ['B-company', 'O', 'O', 'O', 'O', 'B-location', 'O']}
    ]
    entities = [
        Entity(name="company", description="The name of a company"),
        Entity(name="location", description="A physical location"),
    ]

    nlp = spacy.blank("en")
    nlp_config = PipelineConfig(
        linker=LinkerSMXM(),
        entities=entities
    )
    nlp.add_pipe("zshot", config=nlp_config, last=True)
    strategy = ParaphrasingStrategy()
    num_variations = 3

    variations = entropy_heuristic.evaluate_variations_strategy(dataset,
                                                                entities=entities,
                                                                alter_strategy=strategy,
                                                                num_variations=num_variations,
                                                                nlp_pipeline=nlp)

    assert len(variations) == 2
    assert len(variations[0]) == 3 and len(variations[1]) == 3
    assert all([check_is_tuple(x) for x in variations[0]])
    assert all([check_is_tuple(x) for x in variations[1]])
