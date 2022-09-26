from typing import Iterator, List, Optional, Union

import torch
from spacy.tokens import Doc
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from zshot.config import MODELS_CACHE_PATH
from zshot.linker.linker import Linker
from zshot.linker.smxm.data import (
    ByDescriptionTaggerDataset,
    encode_data,
    tagger_multiclass_collator
)
from zshot.linker.smxm.utils import (
    SmxmInput,
    get_entities_names_descriptions,
    load_model,
    predictions_to_span_annotations,
)
from zshot.utils.data_models import Span

SMXM_MODEL_FILES_URL = (
    "https://ibm.box.com/shared/static/duni7p7i4gbk0prksc6zv5uahiemfy00.zip"
)
SMXM_MODEL_FOLDER_NAME = "BertTaggerMultiClass_config03_mode_tagger_multiclass_filtered_classes__entity_descriptions_mode_annotation_guidelines__per_gpu_train_batch_size_7/checkpoint"


class LinkerSMXM(Linker):
    """ SMXM linker """

    def __init__(self):
        super().__init__()

        self.tokenizer = BertTokenizerFast.from_pretrained(
            "bert-large-cased", truncation_side="left"
        )

        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def is_end2end(self) -> bool:
        """ SMXM is end2end model"""
        return True

    def load_models(self):
        """ Load SMXM model """
        if self.model is None:
            self.model = load_model(
                SMXM_MODEL_FILES_URL, MODELS_CACHE_PATH, SMXM_MODEL_FOLDER_NAME
            )

    def predict(self, docs: Iterator[Doc], batch_size: Optional[Union[int, None]] = None) -> List[List[Span]]:
        """
        Perform the entity prediction
        :param docs: A list of spacy Document
        :param batch_size: The batch size
        :return: List Spans for each Document in docs
        """
        if not self._entities:
            return []

        entity_labels, entity_descriptions = get_entities_names_descriptions(self._entities)
        sentences = [doc.text for doc in docs]

        self.load_models()

        encoded_data, max_sentence_tokens = encode_data(
            sentences, entity_labels, entity_descriptions, self.tokenizer
        )
        dataset = ByDescriptionTaggerDataset(encoded_data)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=tagger_multiclass_collator
        )

        preds = []
        probabilities = []
        for batch in dataloader:
            self.model.eval()
            with torch.no_grad():
                inputs = SmxmInput(*batch)
                outputs = self.model(**inputs)
                probability = (
                    torch.nn.Softmax(dim=-1)(outputs).cpu().numpy().tolist()
                )
                probabilities += [p for p in probability]
                outputs = torch.argmax(outputs, dim=2)
                preds += outputs.detach().cpu().numpy().tolist()

        span_annotations = predictions_to_span_annotations(
            sentences, preds, probabilities, entity_labels, self.tokenizer, max_sentence_tokens
        )

        return span_annotations
