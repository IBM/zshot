from typing import List

from datasets import Dataset
from datasets.table import Table

from zshot.utils.data_models import Entity, Relation


class DatasetWithRelations(Dataset):

    def __init__(self, arrow_table: Table, relations: List[Relation] = None, **kwargs):
        super().__init__(arrow_table=arrow_table, **kwargs)
        self.relations = relations

    def __repr__(self):
        return f"Dataset({{\n    features: {list(self.features.keys())},\n    num_rows: {self.num_rows}," \
               f"\n    relations: {[rel.name for rel in self.relations if self.relations is not None]}\n}})"


class DatasetWithEntities(Dataset):

    def __init__(self, arrow_table: Table, entities: List[Entity] = None, **kwargs):
        super().__init__(arrow_table=arrow_table, **kwargs)
        self.entities = entities

    def __repr__(self):
        return f"Dataset({{\n    features: {list(self.features.keys())},\n    num_rows: {self.num_rows}," \
               f"\n    entities: {[ent.name for ent in self.entities if self.entities is not None]}\n}})"


def create_dataset(gt: List[List[str]], sentences: List[str], entities) -> DatasetWithEntities:
    """ Create a simple dataset with entities from sentences and ground truth

    :param gt: Ground truth to use as labels. List of sentences in BIO format
    :param sentences: List of  sentences
    :param entities: List of entities
    :return: Dataset with entities
    """
    data_dict = {
        "tokens": [s.split(" ") for s in sentences],
        "ner_tags": gt,
    }
    dataset = DatasetWithEntities.from_dict(data_dict)
    dataset.entities = entities
    return dataset
