import json
from typing import Dict, Union

from datasets import load_dataset, Split
from huggingface_hub import hf_hub_download

from zshot.evaluation.dataset.dataset import DatasetWithEntities
from zshot.utils.data_models import Entity

REPO_ID = "ibm/medmentionsZS"
ENTITIES_FN = "entities.json"


def load_medmentions() -> Dict[Union[str, Split], DatasetWithEntities]:
    dataset = load_dataset(REPO_ID)
    entities_file = hf_hub_download(repo_id=REPO_ID, repo_type='dataset',
                                    filename=ENTITIES_FN)
    with open(entities_file, "r") as f:
        entities = json.load(f)

    for split in dataset:
        entities_split = [Entity(name=k, description=v) for k, v in entities[split].items()]
        dataset[split] = DatasetWithEntities(dataset[split].data, entities=entities_split)

    return dataset
