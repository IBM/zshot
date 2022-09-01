from typing import Optional, Union, Dict, List

import spacy

from zshot.utils.data_models import Entity
from zshot.linker import Linker
from zshot.mentions_extractor import MentionsExtractor


class PipelineConfig(dict):

    def __init__(self,
                 mentions_extractor: Optional[MentionsExtractor] = None,
                 linker: Optional[Union[Linker, str]] = None,
                 entities: Optional[Union[Dict[str, str], List[Entity], List[str], str]] = None,
                 disable_default_ner: Optional[bool] = True) -> None:
        config = {}

        if mentions_extractor:
            mention_extractor_id = PipelineConfig.param(mentions_extractor)
            config.update({'mentions_extractor': mention_extractor_id})

        if linker:
            linker_id = PipelineConfig.param(linker)
            config.update({'linker': linker_id})

        if entities:
            entities_id = PipelineConfig.param(entities)
            config.update({'entities': entities_id})

        if disable_default_ner:
            config.update({'disable_default_ner': disable_default_ner})

        super().__init__(**config)

    @staticmethod
    def param(param) -> str:
        if isinstance(param, list):
            instance_hash = hash(hash(param[0]) + hash(param[-1]))
        else:
            instance_hash = hash(param)

        instance_id = f"zshot.{param.__class__.__name__}.{instance_hash}"

        @spacy.registry.misc(instance_id)
        def create_custom_component():
            return param

        return instance_id
