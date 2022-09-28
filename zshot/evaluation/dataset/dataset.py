from typing import List, Optional, Dict

import datasets


class DatasetWithEntities(datasets.Dataset):
    # TODO: Implement save and load methods
    def __init__(self, arrow_table: datasets.table.Table,
                 info: Optional[datasets.info.DatasetInfo] = None,
                 split: Optional[datasets.splits.NamedSplit] = None,
                 indices_table: Optional[datasets.table.Table] = None,
                 fingerprint: Optional[str] = None,
                 entities: List[Dict[str, str]] = None):
        super().__init__(arrow_table=arrow_table, info=info, split=split,
                         indices_table=indices_table, fingerprint=fingerprint)
        self.entities = entities

    @classmethod
    def from_dict(
            cls,
            mapping: dict,
            features: Optional[datasets.features.Features] = None,
            info: Optional[datasets.info.DatasetInfo] = None,
            split: Optional[datasets.splits.NamedSplit] = None,
            entities: List[Dict[str, str]] = None
    ) -> "DatasetWithEntities":
        dataset = super().from_dict(mapping=mapping, features=features, info=info, split=split)
        dataset.entities = entities
        return dataset

    def __repr__(self):
        return f"Dataset({{\n    features: {list(self.features.keys())},\n    num_rows: {self.num_rows}," \
               f"\n    entities: {[ent.name for ent in self.entities if self.entities is not None]}\n}})"
