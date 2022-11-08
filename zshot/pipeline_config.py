import random
from typing import Optional, Union, List

import spacy

from zshot.linker import Linker
from zshot.mentions_extractor import MentionsExtractor
from zshot.relation_extractor import RelationsExtractor
from zshot.utils.data_models import Entity, Relation


class PipelineConfig(dict):

    def __init__(self,
                 mentions_extractor: Optional[MentionsExtractor] = None,
                 linker: Optional[Union[Linker, str]] = None,
                 relations_extractor: Optional[Union[RelationsExtractor, str]] = None,
                 mentions: Optional[Union[List[Entity], List[str], str]] = None,
                 entities: Optional[Union[List[Entity], List[str], str]] = None,
                 relations: Optional[Union[List[Relation], str]] = None,
                 disable_default_ner: Optional[bool] = True) -> None:
        config = {}

        if mentions_extractor:
            mention_extractor_id = PipelineConfig.param(mentions_extractor)
            config.update({'mentions_extractor': mention_extractor_id})

        if linker:
            linker_id = PipelineConfig.param(linker)
            config.update({'linker': linker_id})

        if relations_extractor:
            relation_extractor_id = PipelineConfig.param(relations_extractor)
            config.update({'relations_extractor': relation_extractor_id})

        if entities:
            entities_id = PipelineConfig.param(entities)
            config.update({'entities': entities_id})

        if mentions:
            mentions_id = PipelineConfig.param(mentions)
            config.update({'mentions': mentions_id})

        if relations:
            relations_id = PipelineConfig.param(relations)
            config.update({'relations': relations_id})

        if disable_default_ner:
            config.update({'disable_default_ner': disable_default_ner})

        super().__init__(**config)

    @staticmethod
    def param(param) -> str:
        if isinstance(param, list):
            params_to_hash = random.sample(param, k=min(len(param), 10))
            instance_hash = hash(sum([hash(param_to_hash) for param_to_hash in params_to_hash]))
        else:
            instance_hash = hash(param)

        instance_id = f"zshot.{param.__class__.__name__}.{instance_hash}"

        @spacy.registry.misc(instance_id)
        def create_custom_component():
            return param

        return instance_id
