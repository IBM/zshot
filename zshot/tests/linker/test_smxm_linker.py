import os
import shutil

import spacy

from zshot import PipelineConfig
from zshot.config import MODELS_CACHE_PATH
from zshot.linker import LinkerSMXM
from zshot.linker.linker_smxm import SMXM_MODEL_FILES_URL, SMXM_MODEL_FOLDER_NAME
from zshot.linker.smxm.model import BertTaggerMultiClass
from zshot.linker.smxm.utils import load_model
from zshot.tests.config import EX_DOCS, EX_ENTITIES


def test_smxm_download():
    model_folder_path = os.path.join(MODELS_CACHE_PATH, SMXM_MODEL_FOLDER_NAME)
    if os.path.exists(model_folder_path):
        shutil.rmtree(model_folder_path)

    model = load_model(SMXM_MODEL_FILES_URL, MODELS_CACHE_PATH, SMXM_MODEL_FOLDER_NAME)

    assert isinstance(model, BertTaggerMultiClass)


def test_smxm_linker():
    nlp = spacy.blank("en")
    smxm_config = PipelineConfig(
        linker=LinkerSMXM(),
        entities=EX_ENTITIES
    )
    nlp.add_pipe("zshot", config=smxm_config, last=True)
    assert "zshot" in nlp.pipe_names

    doc = nlp(EX_DOCS[1])
    assert len(doc.ents) > 0


def test_smxm_linker_pipeline():
    nlp = spacy.blank("en")
    smxm_config = PipelineConfig(
        linker=LinkerSMXM(),
        entities=EX_ENTITIES
    )
    nlp.add_pipe("zshot", config=smxm_config, last=True)
    assert "zshot" in nlp.pipe_names

    docs = [doc for doc in nlp.pipe(EX_DOCS)]
    assert all(len(doc.ents) > 0 for doc in docs)


def test_smxm_linker_no_entities():
    nlp = spacy.blank("en")
    smxm_config = PipelineConfig(
        linker=LinkerSMXM(),
        entities=[]
    )
    nlp.add_pipe("zshot", config=smxm_config, last=True)
    assert "zshot" in nlp.pipe_names

    doc = nlp(EX_DOCS[1])
    assert len(doc.ents) == 0
