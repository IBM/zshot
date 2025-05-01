from typing import Union, Optional

from datasets import load_dataset

from zshot.evaluation.dataset.dataset import DatasetWithEntities
from zshot.utils.data_models import Entity

REPO_ID = "disi-unibo-nlp/Pile-NER-biomed-IOB"
ENTITIES_REPO_ID = "disi-unibo-nlp/Pile-NER-biomed-descriptions"


def load_pile_ner_biomed_zs(**kwargs) -> DatasetWithEntities:
    dataset = load_dataset(REPO_ID, split='train', **kwargs)
    entities = load_dataset(ENTITIES_REPO_ID, split="train")

    entities_split = [Entity(name=e['entity_type'], description=e['description']) for e in entities]
    dataset = DatasetWithEntities(dataset.data, entities=entities_split)

    return dataset
