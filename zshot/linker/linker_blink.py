import argparse
from pathlib import Path
from typing import List
import pkgutil
from appdata import AppDataPaths
from spacy.tokens import Doc

from zshot.linker.linker import Linker
from zshot.utils import download_file

MODELS_PATH = AppDataPaths(f"{Path(__file__).stem}").app_data_path + "/"

BLINK_FILES = \
    ["http://dl.fbaipublicfiles.com/BLINK/entity.jsonl",
     "http://dl.fbaipublicfiles.com/BLINK/all_entities_large.t7"]
BLINK_BI_ENCODER_FILES = \
    ["http://dl.fbaipublicfiles.com/BLINK/biencoder_wiki_large.json",
     "http://dl.fbaipublicfiles.com/BLINK/biencoder_wiki_large.bin"]
BLINK_CROSS_ENCODER_FILES = \
    ["http://dl.fbaipublicfiles.com/BLINK/crossencoder_wiki_large.bin",
     "http://dl.fbaipublicfiles.com/BLINK/crossencoder_wiki_large.json"]

_config = {
    "test_entities": None,
    "test_mentions": None,
    "interactive": False,
    "top_k": 1,
    "biencoder_model": MODELS_PATH + "biencoder_wiki_large.bin",
    "biencoder_config": MODELS_PATH + "biencoder_wiki_large.json",
    "entity_catalogue": MODELS_PATH + "entity.jsonl",
    "entity_encoding": MODELS_PATH + "all_entities_large.t7",
    "crossencoder_model": MODELS_PATH + "crossencoder_wiki_large.bin",
    "crossencoder_config": MODELS_PATH + "crossencoder_wiki_large.json",
    "fast": True,
    "output_path": "logs/"
}


class Blink(Linker):

    def __init__(self):
        if not pkgutil.find_loader("blink"):
            raise Exception("Blink module not installed. You need to install blink in order to use the Blink Linker."
                            "Install it with: pip install -e git+https://github.com/facebookresearch/BLINK.git#egg"
                            "=BLINK")
        self.config = argparse.Namespace(**_config)
        self.models = None

    def download_models(self):
        for f in BLINK_BI_ENCODER_FILES + BLINK_FILES:
            download_file(f, output_dir=MODELS_PATH)
        if not self.config.fast:
            for f in BLINK_CROSS_ENCODER_FILES:
                download_file(f, output_dir=MODELS_PATH)

    def load_models(self):
        import blink.main_dense as main_dense
        self.download_models()
        if self.models is None:
            self.models = main_dense.load_models(self.config, logger=None)

    def link(self, docs: List[Doc], batch_size=None):
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
        _, _, _, _, _, predictions, scores, = main_dense.run(self.config, None, *self.models, test_data=data_to_link)
        for data, pred in zip(data_to_link, predictions):
            doc = docs[data['id']]
            mention = doc._.mentions[data['mention_id']]
            doc.ents += (doc.char_span(mention.start_char, mention.end_char, label=pred[0]),)
