from typing import Iterator, List, Optional, Union

import pkgutil

from spacy.tokens import Doc
from gliner import GLiNER

from zshot.mentions_extractor.mentions_extractor import MentionsExtractor
from zshot.config import MODELS_CACHE_PATH
from zshot.utils.data_models import Span


MODEL_NAME = "urchade/gliner_mediumv2.1"


class MentionsExtractorGLINER(MentionsExtractor):
    """ GLiNER Mentions Extractor """

    def __init__(self, model_name=MODEL_NAME):
        super().__init__()

        if not pkgutil.find_loader("gliner"):
            raise Exception("GLINER module not installed. You need to install gliner in order to use the GLINER Linker."
                            "Install it with: pip install gliner")

        self.model_name = model_name
        self.model = None

    def load_models(self):
        """ Load GLINER model """
        if self.model is None:
            self.model = GLiNER.from_pretrained(self.model_name, cache_dir=MODELS_CACHE_PATH).to(self.device)
            self.model.eval()

    def predict(self, docs: Iterator[Doc], batch_size: Optional[Union[int, None]] = None) -> List[List[Span]]:
        """
        Perform the entity prediction
        :param docs: A list of spacy Document
        :param batch_size: The batch size
        :return: List Spans for each Document in docs
        """
        if not self._mentions:
            return []

        labels = [ent.name for ent in self._mentions]
        sentences = [doc.text for doc in docs]

        self.load_models()
        span_annotations = []
        for sent in sentences:
            entities = self.model.predict_entities(sent, labels, threshold=0.5)
            span_annotations.append([Span.from_dict(ent) for ent in entities])

        return span_annotations
