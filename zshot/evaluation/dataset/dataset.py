from typing import List

from datasets import Dataset

from zshot.utils.data_models import Entity, Relation


class DatasetWithRelations(Dataset):

    def __init__(self, relations: List[Relation] = None, **kwargs):
        super().__init__(**kwargs)
        self.relations = relations

    def __repr__(self):
        return f"Dataset({{\n    features: {list(self.features.keys())},\n    num_rows: {self.num_rows}," \
               f"\n    entities: {[ent.name for ent in self.relations if self.relations is not None]}\n}})"


class DatasetWithEntities(Dataset):

    def __init__(self, entities: List[Entity] = None, **kwargs):
        super().__init__(**kwargs)
        self.entities = entities

    def __repr__(self):
        return f"Dataset({{\n    features: {list(self.features.keys())},\n    num_rows: {self.num_rows}," \
               f"\n    entities: {[ent.name for ent in self.entities if self.entities is not None]}\n}})"
