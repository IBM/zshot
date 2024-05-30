import numpy as np
import torch
from abc import ABC, abstractmethod
from datasets import Dataset
from spacy import Language
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, BatchEncoding
from typing import List, Tuple, Dict, Optional

from zshot.utils.data_models import Entity


class AlterStrategy(ABC):
    @abstractmethod
    def alter_description(
            self, entity_description: str, num_variations: int
    ) -> List[str]:
        pass


class TransformerAlterStrategy(AlterStrategy):
    def __init__(
            self,
            min_length: int = 80,
            max_length: int = 120,
            num_beams: int = 8,
            no_repeat_ngram_size: int = 2,
            do_sample: bool = True,
            temperature: float = None,
            device: str = None,
    ):
        """ Base class for Alter strategies that use transformers

        :param min_length: Min length of the variations
        :param max_length: Max length of the variations
        :param num_beams: Number of beams for beam search
        :param no_repeat_ngram_size: Parameter for controlling text generation
        :param do_sample: If true use sampling method
        :param temperature: Temperature to use
        :param device: Device to use
        """
        self.min_length = min_length
        self.max_length = max_length
        self.num_beams = num_beams
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.do_sample = do_sample
        self.temperature = temperature
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        print(f"Using device {self.device}")

    def alter_description(self, entity_description: str, num_variations: int) -> List[str]:
        """ Alter the description using the selected strategy

        :param entity_description: Entity description to alter
        :param num_variations: Number of variations to create
        :return: List of description variations
        """
        input_encoding = self._prepare_input(entity_description)
        beam_outputs = self.model.generate(
            inputs=input_encoding["input_ids"].to(self.device),
            attention_mask=input_encoding["attention_mask"].to(self.device),
            pad_token_id=self.tokenizer.eos_token_id,
            min_length=self.min_length,
            max_length=self.max_length,
            num_beams=self.num_beams,
            num_return_sequences=num_variations,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            temperature=self.temperature,
            do_sample=self.do_sample,
        )
        new_descriptions = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in beam_outputs
        ]
        return new_descriptions

    def _get_initial_description(self, entity_description: str) -> str:
        """ Get the initial description of the entity

        :param entity_description: Tokenized entity description
        :return:
        """
        return " ".join(entity_description.split()[:10])

    @abstractmethod
    def _prepare_input(self, entity_description: str) -> BatchEncoding:
        """ Prepare the input for the alter strategy

        :param entity_description: Description of the entity to alter
        :return:
        """
        pass


class PreTrainedLMExtensionStrategy(TransformerAlterStrategy):
    def __init__(self, model_name_or_path: Optional[str] = "gpt2"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.model.to(self.device)

    def _prepare_input(self, entity_description: str) -> BatchEncoding:
        """ Prepare the input for the alter strategy

        :param entity_description: Description of the entity to alter
        :return:
        """
        initial_description = self._get_initial_description(entity_description)
        return self.tokenizer(initial_description, return_tensors="pt")


class FineTunedLMExtensionStrategy(TransformerAlterStrategy):
    def __init__(self, model_name_or_path: Optional[str] = "lfuchs/desctension"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.model.to(self.device)

    def _prepare_input(self, entity_description: str) -> BatchEncoding:
        """ Prepare the input for the alter strategy

        :param entity_description: Description of the entity to alter
        :return:
        """
        initial_description = self._get_initial_description(entity_description)
        return self.tokenizer(
            f"extend description: {initial_description}", return_tensors="pt"
        )


class SummarizationStrategy(TransformerAlterStrategy):
    def __init__(self,
                 model_name_or_path: Optional[
                     str] = "mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.model.to(self.device)

    def _prepare_input(self, entity_description: str) -> BatchEncoding:
        """ Prepare the input for the alter strategy

        :param entity_description: Description of the entity to alter
        :return:
        """
        return self.tokenizer(entity_description, padding="max_length", truncation=True,
                              max_length=512, return_tensors="pt")


class ParaphrasingStrategy(TransformerAlterStrategy):
    def __init__(self, model_name_or_path: Optional[str] = "tuner007/pegasus_paraphrase"):
        super().__init__(min_length=10, max_length=60, temperature=1.5)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.model.to(self.device)

    def _prepare_input(self, entity_description: str) -> BatchEncoding:
        """ Prepare the input for the alter strategy

        :param entity_description: Description of the entity to alter
        :return:
        """
        return self.tokenizer(entity_description, truncation=True, padding='longest',
                              max_length=60, return_tensors="pt")


class DescriptionHeuristic(ABC):
    @abstractmethod
    def evaluate_variations_strategy(
            self,
            dataset: Dataset,
            entities: List[Entity],
            alter_strategy: AlterStrategy,
            num_variations: int,
            nlp_pipeline: Language,
    ) -> List[List[Tuple[str, float]]]:
        """ Evaluate all the variations of all entities over a dataset

        :param dataset: Dataset to use for evaluation
        :param entities: List of entities of the dataset
        :param alter_strategy: Strategy used to create the variations
        :param num_variations: Number of variations to create
        :param nlp_pipeline: Spacy NLP pipeline
        :param is_only_negative: True for using only the focus entity. False for use all the entities in the pipeline
        :param batch_size: Batch size to use
        :return: List of tuple with each variation and the heuristic result
        """
        pass

    @abstractmethod
    def evaluate_variations_strategy_for_entity(
            self,
            dataset: Dataset,
            entities: List[Entity],
            focus_entity: Entity,
            alter_strategy: AlterStrategy,
            num_variations: int,
            nlp_pipeline: Language,
    ) -> List[Tuple[str, float]]:
        """ Evaluate all the variations of an entity over a dataset

        :param dataset: Dataset to use for evaluation
        :param entities: List of entities of the dataset
        :param alter_strategy: Strategy used to create the variations
        :param num_variations: Number of variations to create
        :param nlp_pipeline: Spacy NLP pipeline
        :param is_only_negative: True for using only the focus entity. False for use all the entities in the pipeline
        :param batch_size: Batch size to use
        :return: List of tuple with each variation and the heuristic result
        """
        pass


class EntropyHeuristic(DescriptionHeuristic):
    def evaluate_variations_strategy(
            self,
            dataset: Dataset,
            entities: List[Entity],
            alter_strategy: AlterStrategy,
            num_variations: int,
            nlp_pipeline: Language,
            is_only_negative: bool = False,
            batch_size: int = 8,
    ) -> List[List[Tuple[str, float]]]:
        """ Evaluate all the variations of all entities over a dataset

        :param dataset: Dataset to use for evaluation
        :param entities: List of entities of the dataset
        :param alter_strategy: Strategy used to create the variations
        :param num_variations: Number of variations to create
        :param nlp_pipeline: Spacy NLP pipeline
        :param is_only_negative: True for using only the focus entity. False for use all the entities in the pipeline
        :param batch_size: Batch size to use
        :return: List of tuple with each variation and the entropy result
        """

        eval_variation = []
        for entity in entities:
            eval_var = self.evaluate_variations_strategy_for_entity(
                dataset,
                entities,
                entity,
                alter_strategy,
                num_variations,
                nlp_pipeline,
                is_only_negative,
                batch_size
            )
            eval_variation.append(eval_var)
        return eval_variation

    def evaluate_variations_strategy_for_entity(
            self,
            dataset: Dataset,
            entities: List[Entity],
            focus_entity: Entity,
            alter_strategy: AlterStrategy,
            num_variations: int,
            nlp_pipeline: Language,
            is_only_negative: Optional[bool] = False,
            batch_size: Optional[int] = 8,
    ) -> List[Tuple[str, float]]:
        """ Evaluate all the variations of an entity over a dataset

        :param dataset: Dataset to use for evaluation
        :param entities: List of entities of the dataset
        :param focus_entity: Entity to evaluate
        :param alter_strategy: Strategy used to create the variations
        :param num_variations: Number of variations to create
        :param nlp_pipeline: Spacy NLP pipeline
        :param is_only_negative: True for using only the focus entity. False for use all the entities in the pipeline
        :param batch_size: Batch size to use
        :return: List of tuple with each variation and the entropy result
        """

        def collect_batch(s_batch: List[Dict]) -> List[str]:
            return list(map(lambda b: " ".join(b["tokens"]).strip(), s_batch))

        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collect_batch)
        desc_variations = alter_strategy.alter_description(
            focus_entity.description, num_variations=num_variations
        )
        all_entities = set(entities) - {focus_entity}
        variations_scores = []

        for desc_variation in desc_variations:
            batches_entropy = []
            for batch in dataloader:
                variation_entity = Entity(name=focus_entity.name, description=desc_variation)
                if is_only_negative:
                    used_entities = {variation_entity}
                else:
                    used_entities = all_entities.union({variation_entity})
                nlp_pipeline.get_pipe("zshot").entities = list(used_entities)
                batches_entropy.extend(
                    [
                        list(map(lambda s: s.score, doc._.spans))
                        for doc in nlp_pipeline.pipe(batch)
                    ]
                )
            variations_scores.append(self.calculate_entropy(batches_entropy))
        return list(zip(desc_variations, variations_scores))

    @staticmethod
    def calculate_entropy(sentences_predictions: List[List[float]]) -> float:
        """
        Calculate the entropy of predictions.
        :param sentences_predictions: Probability distribution for classes in tokens in sentences.
        :return: Entropy of the probability distribution.
        """
        entropies = []
        for probs in sentences_predictions:
            lp = len(probs)
            score = -np.sum(probs * np.log2(probs) + (np.ones(lp) - probs) * np.log2((np.ones(lp) - probs))) / lp
            if not np.isnan(score):
                entropies.append(score)
        return float(np.mean(entropies))
