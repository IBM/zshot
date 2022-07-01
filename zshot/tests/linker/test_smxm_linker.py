import os
import spacy
from appdata import AppDataPaths
import shutil

from zshot import PipelineConfig
from zshot.linker import LinkerSMXM
from zshot.linker.smxm.model import BertTaggerMultiClass
from zshot.linker.smxm.utils import load_model
from zshot.tests.config import EX_DOCS, EX_ENTITIES


def test_smxm_download():
    MODELS_CACHE_PATH = (
        os.getenv("SMXM_MODELS_CACHE_PATH")
        if "SMXM_MODELS_CACHE_PATH" in os.environ
        else AppDataPaths("linker_smxm").app_data_path + "/"
    )
    MODEL_FILES_URL = "https://ibm.box.com/shared/static/uqav0794cbfrzru2q3seru0xm7pz336c.zip"
    MODEL_FOLDER_NAME = "BertTaggerMultiClass_config03_mode_tagger_multiclass_filtered_classes__entity_descriptions_mode_annotation_guidelines__per_gpu_train_batch_size_7/checkpoint"

    if os.path.exists(MODELS_CACHE_PATH):
        shutil.rmtree(MODELS_CACHE_PATH)

    model = load_model(MODEL_FILES_URL, MODELS_CACHE_PATH, MODEL_FOLDER_NAME)

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
