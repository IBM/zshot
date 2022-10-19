import argparse
import os
import pkgutil
from enum import Enum
from typing import List, Iterator, Optional, Union

from spacy.tokens import Doc

from zshot.config import MODELS_CACHE_PATH
from zshot.linker.linker import Linker
from zshot.utils.data_models import Span
from zshot.utils.file_utils import download_file

BLINK_ENTITIES = "http://dl.fbaipublicfiles.com/BLINK/entity.jsonl"

BLINK_ENTITIES_ENCODING = "http://dl.fbaipublicfiles.com/BLINK/all_entities_large.t7"

BLINK_BI_ENCODER_FILES = \
    ["http://dl.fbaipublicfiles.com/BLINK/biencoder_wiki_large.json",
     "http://dl.fbaipublicfiles.com/BLINK/biencoder_wiki_large.bin"]
BLINK_CROSS_ENCODER_FILES = \
    ["http://dl.fbaipublicfiles.com/BLINK/crossencoder_wiki_large.bin",
     "http://dl.fbaipublicfiles.com/BLINK/crossencoder_wiki_large.json"]

BLINK_FAISS_FLAT_INDEX = "http://dl.fbaipublicfiles.com/BLINK//faiss_flat_index.pkl"
BLINK_FAISS_HNSW_INDEX = "http://dl.fbaipublicfiles.com/BLINK/faiss_hnsw_index.pkl"

_config = {
    "test_entities": None,
    "test_mentions": None,
    "interactive": False,
    "top_k": 1,
    "biencoder_model": MODELS_CACHE_PATH + "biencoder_wiki_large.bin",
    "biencoder_config": MODELS_CACHE_PATH + "biencoder_wiki_large.json",
    "entity_catalogue": MODELS_CACHE_PATH + "entity.jsonl",
    "entity_encoding": MODELS_CACHE_PATH + "all_entities_large.t7",
    "crossencoder_model": MODELS_CACHE_PATH + "crossencoder_wiki_large.bin",
    "crossencoder_config": MODELS_CACHE_PATH + "crossencoder_wiki_large.json",
    "fast": True,
    "faiss_index": None,
    "index_path": None,
    "output_path": "logs/"
}


class BlinkIndex(str, Enum):
    FLAT = "flat"
    HNSW = "hnsw"
    NONE = None


blink_index2url = {BlinkIndex.FLAT: BLINK_FAISS_FLAT_INDEX, BlinkIndex.HNSW: BLINK_FAISS_HNSW_INDEX}


class LinkerBlink(Linker):
    """ Blink linker """

    def __init__(self, index=BlinkIndex.FLAT):
        """
        :param index: Index to use to perform the entity linking. One of:
            - BlinkIndex.FLAT
            - BlinkIndex.HNSW
        """
        super().__init__()

        if not pkgutil.find_loader("blink"):
            raise Exception("Blink module not installed. You need to install blink in order to use the Blink Linker."
                            "Install it with: pip install -e git+https://github.com/facebookresearch/BLINK.git#egg"
                            "=BLINK")
        self.index = index
        if index:
            _config['faiss_index'] = index.value
            _config['index_path'] = os.path.join(MODELS_CACHE_PATH, blink_index2url[index].split('/')[-1])
        self.config = argparse.Namespace(**_config)
        self.models = None
        self._wikipedia_id2local_id = None

    @property
    def entities_list(self) -> List[str]:
        """ Get list of entities """
        if len(self.models) < 6:
            raise Exception('model not yet initialized')
        return list(self.models[5].keys())

    @property
    def local_id2wikipedia_id(self):
        """ Get the Wikipedia ID from the label predicted """
        if len(self.models) < 9:
            raise Exception('model not yet initialized')
        if not self._wikipedia_id2local_id:
            self._wikipedia_id2local_id = {v: k for k, v in self.models[8].items()}
        return self._wikipedia_id2local_id

    def local_name2wikipedia_url(self, label: str) -> str:
        """ Get the Wikipedia URL of the label predicted to perform wikification

        :param label: Label to get URL of
        :return: URL of the Wikipedia article
        """
        wiki_id = self.local_id2wikipedia_id[self.models[5][label]]
        return f"https://en.wikipedia.org/wiki?curid={wiki_id}"

    def download_models(self):
        """ Download Blink files """
        files_to_download = BLINK_BI_ENCODER_FILES + [BLINK_ENTITIES]
        if self.index:
            files_to_download.append(blink_index2url[self.index])
        else:
            files_to_download.append(BLINK_ENTITIES_ENCODING)
        for f in files_to_download:
            download_file(f, output_dir=MODELS_CACHE_PATH)
        if not self.config.fast:
            for f in BLINK_CROSS_ENCODER_FILES:
                download_file(f, output_dir=MODELS_CACHE_PATH)

    def load_models(self):
        """ Load models """
        import blink.main_dense as main_dense
        if self.models is None:
            self.download_models()
            self.models = main_dense.load_models(self.config, logger=None)

    def predict(self, docs: Iterator[Doc], batch_size: Optional[Union[int, None]] = None) -> List[List[Span]]:
        """
        Perform the entity prediction
        :param docs: A list of spacy Document
        :param batch_size: The batch size
        :return: List Spans for each Document in docs
        """
        import blink.main_dense as main_dense
        self.load_models()
        data_to_link = []
        for doc_id, doc in enumerate(docs):
            for mention_id, mention in enumerate(doc._.mentions):
                left_context = doc.text[:mention.start].lower()
                right_context = doc.text[mention.end:].lower()
                text = doc.text[mention.start:mention.end].lower()
                data_to_link.append(
                    {
                        "id": doc_id,
                        "mention_id": mention_id,
                        "label": "unknown",
                        "label_id": -1,
                        "context_left": left_context,
                        "mention": text,
                        "context_right": right_context,
                    })
        if not data_to_link:
            return []

        _, _, _, _, _, predictions, scores, = main_dense.run(self.config, None, *self.models, test_data=data_to_link)
        spans_annotations_values = {}
        for data, pred, score in zip(data_to_link, predictions, scores):
            doc = docs[data['id']]
            mention = doc._.mentions[data['mention_id']]
            span = Span(mention.start, mention.end, label=pred[0], score=score,
                        kb_id=self.local_name2wikipedia_url(pred[0]))
            if data['id'] in spans_annotations_values:
                spans_annotations_values['id'].append(span)
            else:
                spans_annotations_values['id'] = [span]

        spans_annotations_keys = sorted(spans_annotations_values.keys())
        spans_annotations = [spans_annotations_values[key] for key in spans_annotations_keys]

        return spans_annotations
