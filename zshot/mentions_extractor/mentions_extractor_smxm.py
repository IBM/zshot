from typing import Iterator, List, Optional, Union

import torch
from spacy.tokens import Doc
from transformers import BertTokenizerFast

from zshot.mentions_extractor.mentions_extractor import MentionsExtractor
from zshot.utils.models.smxm.model import BertTaggerMultiClass, device
from zshot.utils.models.smxm.utils import (
    get_entities_names_descriptions,
    smxm_predict,
)
from zshot.utils.data_models import Span

MODEL_NAME = "ibm/smxm"


class MentionsExtractorSMXM(MentionsExtractor):
    """ SMXM Mentions Extractor """

    def __init__(self):
        super().__init__()

        self.tokenizer = BertTokenizerFast.from_pretrained(
            "bert-large-cased", truncation_side="left"
        )

        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_models(self):
        """ Load SMXM model """
        if self.model is None:
            self.model = BertTaggerMultiClass.from_pretrained(
                MODEL_NAME, output_hidden_states=True
            ).to(device)

    def predict(self, docs: Iterator[Doc], batch_size: Optional[Union[int, None]] = None) -> List[List[Span]]:
        """
        Perform the entity prediction
        :param docs: A list of spacy Document
        :param batch_size: The batch size
        :return: List Spans for each Document in docs
        """
        if not self._mentions:
            return []

        entity_labels, entity_descriptions = get_entities_names_descriptions(self._mentions)
        sentences = [doc.text for doc in docs]

        self.load_models()
        self.model.eval()

        span_annotations = smxm_predict(self.model, self.tokenizer,
                                        sentences, entity_labels, entity_descriptions,
                                        batch_size)

        return span_annotations
