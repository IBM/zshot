import json
import re
from typing import Union, Optional

from datasets import load_dataset, Split, Dataset, DatasetDict
from huggingface_hub import hf_hub_download

from zshot.evaluation.dataset.dataset import DatasetWithEntities
from zshot.utils.data_models import Entity

REPO_ID = "ibm/MedMentions-ZS"
ENTITIES_FN = "entities.json"


def load_medmentions_zs(split: Optional[Union[str, Split]] = None, **kwargs) -> Union[DatasetDict, Dataset]:
    dataset = load_dataset(REPO_ID, split=split, **kwargs)
    entities_file = hf_hub_download(repo_id=REPO_ID,
                                    repo_type='dataset',
                                    filename=ENTITIES_FN)
    with open(entities_file, "r") as f:
        entities = json.load(f)

    if split:
        entities_split = [Entity(name=k, description=v) for k, v in entities[get_simple_split(split)].items()]
        dataset = DatasetWithEntities(dataset.data, entities=entities_split)
    else:
        for split in dataset:
            entities_split = [Entity(name=k, description=v) for k, v in entities[split].items()]
            dataset[split] = DatasetWithEntities(dataset[split].data, entities=entities_split)

    return dataset


def get_simple_split(split: str) -> str:
    first_not_alph = re.search(r'\W+', split)
    first_not_alph_chr = first_not_alph.start() if first_not_alph else len(split)
    return split[: first_not_alph_chr]
