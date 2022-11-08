import itertools
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import spacy
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from zshot import PipelineConfig
from zshot.utils.data_models import Entity
from zshot.linker import LinkerSMXM

Strategy = Enum("Strategy", ["ONLY_NEG", "INITIAL_DESCRIPTIONS", "ALL_COMBINATIONS"])
StartingPoint = Enum(
    "StartingPoint", ["INITIAL_DESCRIPTION", "JUST_NAME", "CAN_BE_DEFINED_AS"]
)


class DescriptionEnrichment:
    """
    Provides functionality to enhance given type descriptions with different strategies.
    """
    def __init__(
        self,
        test_sentences: List[str],
        model_id: str,
        max_length: int,
        no_repeat_ngram_size: Optional[int] = 0,
        do_sample: Optional[bool] = True,
        early_stopping: Optional[bool] = True,
        clean_text: Optional[bool] = False,
        starting_point: Optional[StartingPoint] = StartingPoint.INITIAL_DESCRIPTION,
        is_seq2seq: Optional[bool] = False,
    ):
        self.test_sentences = test_sentences
        self.model_id = model_id
        self.is_seq2seq = is_seq2seq
        self.max_length = max_length
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.do_sample = do_sample
        self.early_stopping = early_stopping
        self.clean_text = clean_text
        if starting_point in StartingPoint.__members__.values() and not (
            self.is_seq2seq and (starting_point == StartingPoint.CAN_BE_DEFINED_AS)
        ):
            self.starting_point = starting_point
        else:
            raise ValueError("Please choose a valid starting point.")

        try:
            if self.is_seq2seq:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        except EnvironmentError:
            print("Please provide a valid path or huggingface model_id!")

    def get_new_description_candidates(
        self,
        entities: List[Entity],
        num_candidates: int,
        entities_test_strategy: Strategy,
    ) -> List[Dict[str, Any]]:
        """
        Get enhanced descriptions for a list of entities.
        :param entities: List of entities.
        :param num_candidates: Number of description candidates to be generated for each entity.
        :param entities_test_strategy: Enhancement strategy to apply.
        :return: List of entities with enhanced descriptions.
        """
        if entities_test_strategy not in Strategy.__members__.values():
            raise ValueError("Please choose a valid strategy.")

        candidates_per_entity = [
            {
                "entity": entity.name,
                "candidates": self.extend_description(entity, num_candidates),
            }
            for entity in entities
        ]

        if entities_test_strategy != Strategy.ALL_COMBINATIONS:
            for i, entity in enumerate(entities):

                if entities_test_strategy == Strategy.ONLY_NEG:
                    candidates = [
                        {
                            **candidate,
                            "probability_distribution": self.get_probability_distribution(
                                [
                                    Entity(
                                        name=entity.name,
                                        description=candidate["description"],
                                    )
                                ]
                            ),
                        }
                        for candidate in candidates_per_entity[i]["candidates"]
                    ]

                elif entities_test_strategy == Strategy.INITIAL_DESCRIPTIONS:
                    candidates = [
                        {
                            **candidate,
                            "probability_distribution": self.get_probability_distribution(
                                [
                                    Entity(
                                        name=entity.name,
                                        description=candidate["description"],
                                    )
                                    if j == i
                                    else ent
                                    for j, ent in enumerate(entities)
                                ]
                            ),
                        }
                        for candidate in candidates_per_entity[i]["candidates"]
                    ]

                candidates_per_entity[i]["candidates"] = candidates

        else:
            all_possible_descriptions = [[entity.description] for entity in entities]
            for i, entity in enumerate(candidates_per_entity):
                for j, candidate in enumerate(entity["candidates"]):
                    candidates_per_entity[i]["candidates"][j][
                        "probability_distribution"
                    ] = []
                    all_possible_descriptions[i].append(candidate["description"])

            all_combinations = list(
                itertools.product(
                    *[
                        [i for i in range(len(entities) + 1)]
                        for j in range(len(entities))
                    ]
                )
            )

            for combination in all_combinations:
                prod_dist = self.get_probability_distribution(
                    [
                        Entity(
                            name=entities[i].name,
                            description=all_possible_descriptions[i][comb_idx],
                        )
                        for i, comb_idx in enumerate(combination)
                    ]
                )
                for i, comb_idx in enumerate(combination):
                    if comb_idx != 0:
                        candidates_per_entity[i]["candidates"][comb_idx - 1][
                            "probability_distribution"
                        ] += prod_dist

        for i in range(len(entities)):
            candidates = [
                {
                    **candidate,
                    "entropy": self.calculate_entropy(
                        candidate["probability_distribution"]
                    ),
                }
                for candidate in candidates_per_entity[i]["candidates"]
            ]
            candidates_per_entity[i]["candidates"] = candidates

        sorted_candidates_per_entity = [
            {
                **entity,
                **{
                    "candidates": sorted(
                        entity["candidates"], key=lambda c: c["entropy"]
                    )
                },
            }
            for entity in candidates_per_entity
        ]

        return sorted_candidates_per_entity

    def extend_description(
        self, entity: Entity, num_candidates: int
    ) -> List[Dict[str, str]]:
        """
        Get enhanced descriptions for a single entity.
        :param entity: The entity that should be taken into account.
        :param num_candidates: Number of description candidates to be generated for the entity.
        :return: List of enhanced descriptions.
        """
        if self.is_seq2seq:
            if self.starting_point == StartingPoint.JUST_NAME:
                input_encoding = self.tokenizer(
                    f"generate description: {entity.name}", return_tensors="pt"
                )
            else:
                input_encoding = self.tokenizer(
                    f"extend description: {entity.description}", return_tensors="pt"
                )
        else:
            if self.starting_point == StartingPoint.JUST_NAME:
                input_encoding = self.tokenizer(entity.name, return_tensors="pt")
            elif self.starting_point == StartingPoint.CAN_BE_DEFINED_AS:
                input_encoding = self.tokenizer(
                    f"{entity.name} can be defined as", return_tensors="pt"
                )
            else:
                input_encoding = self.tokenizer(entity.description, return_tensors="pt")

        beam_outputs = self.model.generate(
            inputs=input_encoding["input_ids"],
            attention_mask=input_encoding["attention_mask"],
            pad_token_id=self.tokenizer.eos_token_id,
            max_length=self.max_length,
            num_beams=num_candidates,
            num_return_sequences=num_candidates,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            do_sample=self.do_sample,
            early_stopping=self.early_stopping,
        )

        new_descriptions = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in beam_outputs
        ]

        if self.clean_text:
            for i, d in enumerate(new_descriptions):
                d = d.replace("\n\n", " ").replace("\n", " ")
                if d[-1] != ".":
                    d = d[: d.rfind(". ") + 1]
                if len(d) > 0:
                    new_descriptions[i] = d

        return [{"description": description} for description in new_descriptions]

    def get_probability_distribution(
        self, entities: List[Entity]
    ) -> List[List[List[float]]]:
        """
        Get the probability distribution of test predictions for a list of entity descriptions.
        :param entities: List of entities.
        :return: Probability distribution of test predictions.
        """
        zshot_config = PipelineConfig(
            linker=LinkerSMXM(),
            entities=entities,
        )
        nlp = spacy.blank("en")
        nlp.add_pipe("zshot", config=zshot_config, last=True)

        return [doc._.probability for doc in nlp.pipe(self.test_sentences)]

    def calculate_entropy(self, prob_dist: List[List[List[float]]]) -> float:
        """
        Calculate the entropy of a probability distribution.
        :param prob_dist: Probability distribution for classes in tokens in sentences.
        :return: Entropy of the probability distribution.
        """
        entropies = []
        for sentence_prob_dist in prob_dist:
            for token_prob_dist in sentence_prob_dist:
                entropies.append(-np.sum(token_prob_dist * np.log2(token_prob_dist)))

        return np.mean(entropies)
