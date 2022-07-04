import os
from typing import Any, Dict, Iterator, List, Optional, Union

import torch
from appdata import AppDataPaths
from spacy.tokens import Doc
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from zshot.entity import Entity
from zshot.linker.linker import Linker
from zshot.linker.smxm.data import (ByDescriptionTaggerDataset, encode_data,
                                    tagger_multiclass_collator)
from zshot.linker.smxm.utils import (SmxmInput,
                                     get_entities_names_descriptions,
                                     load_model,
                                     predictions_to_span_annotations)

MODELS_CACHE_PATH = (
    os.getenv("MODELS_CACHE_PATH")
    if "MODELS_CACHE_PATH" in os.environ
    else AppDataPaths("linker_smxm").app_data_path + "/"
)
SMXM_MODEL_FILES_URL = (
    "https://ibm.box.com/shared/static/uqav0794cbfrzru2q3seru0xm7pz336c.zip"
)
SMXM_MODEL_FOLDER_NAME = "BertTaggerMultiClass_config03_mode_tagger_multiclass_filtered_classes__entity_descriptions_mode_annotation_guidelines__per_gpu_train_batch_size_7/checkpoint"


class LinkerSMXM(Linker):

    def __init__(self):
        super().__init__()

        self.tokenizer = BertTokenizerFast.from_pretrained("bert-large-cased")
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def is_end2end(self) -> bool:
        return True

    def link(self, docs: Iterator[Doc], batch_size: Optional[Union[int, None]] = None):
        if not self._entities:
            return

        predictions = self.predict(docs, self._entities, batch_size)
        for doc, doc_preds in zip(docs, predictions):
            for pred in doc_preds:
                doc.ents += (
                    doc.char_span(pred["start"], pred["end"], label=pred["label"]),
                )

    def load_models(self):
        if self.model is None:
            self.model = load_model(
                SMXM_MODEL_FILES_URL, MODELS_CACHE_PATH, SMXM_MODEL_FOLDER_NAME
            )

    def predict(
        self, docs: Iterator[Doc], entities: List[Entity], batch_size: Union[int, None]
    ) -> List[List[Dict[str, Any]]]:

        entity_labels, entity_descriptions = get_entities_names_descriptions(entities)
        sentences = [doc.text for doc in docs]

        self.load_models()

        encoded_data = encode_data(
            sentences, entity_labels, entity_descriptions, self.tokenizer
        )
        dataset = ByDescriptionTaggerDataset(encoded_data)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=tagger_multiclass_collator
        )

        preds = []
        for batch in dataloader:
            self.model.eval()
            with torch.no_grad():
                inputs = SmxmInput(*batch)
                outputs, _ = self.model(**inputs)
                outputs = torch.argmax(outputs, dim=2)
                preds += outputs.detach().cpu().numpy().tolist()

        span_annotations = predictions_to_span_annotations(
            sentences, preds, entity_labels, self.tokenizer
        )

        return span_annotations
