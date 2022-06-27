import os
import pkgutil
from typing import Any, Dict, Iterator, List, Optional, Union

from appdata import AppDataPaths
from spacy.tokens import Doc

from zshot.entity import Entity
from zshot.linker.linker import Linker

MODELS_CACHE_PATH = (
    os.getenv("SMXM_MODELS_CACHE_PATH")
    if "SMXM_MODELS_CACHE_PATH" in os.environ
    else AppDataPaths("linker_smxm").app_data_path + "/"
)
MODEL_FILES_URL = "https://drive.google.com/uc?id=1PGEyBsuc6n085j9kZ5TtkAV7hC5mggdd"
MODEL_FOLDER_NAME = "BertTaggerMultiClass_config03_mode_tagger_multiclass_filtered_classes__entity_descriptions_mode_annotation_guidelines__per_gpu_train_batch_size_7/checkpoint"


class LinkerSMXM(Linker):
    def __init__(self):
        super().__init__()

        if not pkgutil.find_loader("torch"):
            raise Exception(
                "Torch module not installed. You need to install Torch for using this class."
                "Install it with: pip install torch==1.11.0"
            )
        if not pkgutil.find_loader("transformers"):
            raise Exception(
                "Transformers module not installed. You need to install Transformers for using this class."
                "Install it with: pip install transformers==4.19.4"
            )
        if not pkgutil.find_loader("gdown"):
            raise Exception(
                "Gdown module not installed. You need to install Gdown for using this class."
                "Install it with: pip install gdown==4.4.0"
            )

        import torch
        from transformers import BertTokenizerFast

        self.tokenizer = BertTokenizerFast.from_pretrained("bert-large-cased")
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def is_end2end(self) -> bool:
        return True

    def link(self, docs: Iterator[Doc], batch_size: Optional[Union[int, None]] = None):
        if not self._entities:
            return
        if not any(e.name == "NEG" for e in self._entities):
            raise Exception("The negative entity with the name 'NEG' must be provided.")

        predictions = self.predict(docs, self._entities, batch_size)
        for doc, doc_preds in zip(docs, predictions):
            for pred in doc_preds:
                doc.ents += (
                    doc.char_span(pred["start"], pred["end"], label=pred["label"]),
                )

    def predict(
        self, docs: Iterator[Doc], entities: List[Entity], batch_size: Union[int, None]
    ) -> List[List[Dict[str, Any]]]:
        import torch
        from torch.utils.data import DataLoader

        from zshot.linker.smxm.data import (ByDescriptionTaggerDataset,
                                            encode_data,
                                            tagger_multiclass_collator)
        from zshot.linker.smxm.utils import (SmxmInput, load_model,
                                             predictions_to_span_annotations)

        neg_index = [e.name for e in entities].index("NEG")
        entities.insert(0, entities.pop(neg_index))
        entity_labels = [e.name for e in entities]
        entity_descriptions = [e.description for e in entities]
        sentences = [doc.text for doc in docs]

        if self.model is None:
            self.model = load_model(
                MODEL_FILES_URL, MODELS_CACHE_PATH, MODEL_FOLDER_NAME
            )

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
