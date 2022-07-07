import argparse
import os
import pkgutil
from enum import Enum
from pathlib import Path
from typing import List, Iterator

from appdata import AppDataPaths
from spacy.tokens import Doc

from zshot.linker.linker import Linker
from zshot.utils.utils import download_file

MODELS_CACHE_PATH = os.getenv("MODELS_CACHE_PATH") if "MODELS_CACHE_PATH" in os.environ \
    else AppDataPaths(f"{Path(__file__).stem}").app_data_path + "/"

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

    def __init__(self, index=BlinkIndex.FLAT):
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
        if len(self.models) < 6:
            raise Exception('model not yet initialized')
        return list(self.models[5].keys())

    @property
    def local_id2wikipedia_id(self):
        if len(self.models) < 9:
            raise Exception('model not yet initialized')
        if not self._wikipedia_id2local_id:
            self._wikipedia_id2local_id = {v: k for k, v in self.models[8].items()}
        return self._wikipedia_id2local_id

    def local_name2wikipedia_url(self, label: str) -> str:
        wiki_id = self.local_id2wikipedia_id[self.models[5][label]]
        return f"https://en.wikipedia.org/wiki?curid={wiki_id}"

    def download_models(self):
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
        import blink.main_dense as main_dense
        if self.models is None:
            self.download_models()
            self.models = main_dense.load_models(self.config, logger=None)

    def link(self, docs: Iterator[Doc], batch_size=None):
        import blink.main_dense as main_dense
        self.load_models()
        data_to_link = []
        for doc_id, doc in enumerate(docs):
            for mention_id, mention in enumerate(doc._.mentions):
                data_to_link.append(
                    {
                        "id": doc_id,
                        "mention_id": mention_id,
                        "label": "unknown",
                        "label_id": -1,
                        "context_left": doc.text[:mention.start_char].lower(),
                        "mention": mention.text.lower(),
                        "context_right": doc.text[mention.end_char:].lower(),
                    })
        if not data_to_link:
            return
        _, _, _, _, _, predictions, scores, = main_dense.run(self.config, None, *self.models, test_data=data_to_link)
        for data, pred in zip(data_to_link, predictions):
            doc = docs[data['id']]
            mention = doc._.mentions[data['mention_id']]
            doc.ents += (doc.char_span(mention.start_char, mention.end_char, label=pred[0],
                                       kb_id=self.local_name2wikipedia_url(pred[0])))