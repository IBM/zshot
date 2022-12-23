from typing import Iterator, List, Optional, Union

from spacy.tokens import Doc
from transformers import BertTokenizerFast

from zshot.linker.linker import Linker
from zshot.utils.data_models import Span
from zshot.utils.models.smxm.model import BertTaggerMultiClass
from zshot.utils.models.smxm.utils import (
    get_entities_names_descriptions,
    smxm_predict
)

ONTONOTES_MODEL_NAME = "ibm/smxm"


class LinkerSMXM(Linker):
    """ SMXM linker """

    def __init__(self, model_name=ONTONOTES_MODEL_NAME):
        super().__init__()

        self.tokenizer = BertTokenizerFast.from_pretrained(
            "bert-large-cased", truncation_side="left"
        )

        self.model_name = model_name
        self.model = None

    @property
    def is_end2end(self) -> bool:
        """ SMXM is end2end model"""
        return True

    def load_models(self):
        """ Load SMXM model """
        if self.model is None:
            self.model = BertTaggerMultiClass.from_pretrained(
                self.model_name, output_hidden_states=True
            ).to(self.device)

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
        self.model.eval()

        span_annotations = smxm_predict(self.model, self.tokenizer,
                                        sentences, entity_labels, entity_descriptions,
                                        batch_size)

        return span_annotations
