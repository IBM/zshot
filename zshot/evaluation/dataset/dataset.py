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
               f"\n    entities: {[ent.name for ent in self.relations if self.relations is not None]}\n}})"


class DatasetWithEntities(Dataset):

    def __init__(self, arrow_table: Table, entities: List[Entity] = None, **kwargs):
        super().__init__(arrow_table=arrow_table, **kwargs)
        self.entities = entities

    def __repr__(self):
        return f"Dataset({{\n    features: {list(self.features.keys())},\n    num_rows: {self.num_rows}," \
               f"\n    entities: {[ent.name for ent in self.entities if self.entities is not None]}\n}})"
