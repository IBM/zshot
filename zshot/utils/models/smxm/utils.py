from functools import partial
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from zshot.utils.models.smxm.data import encode_data, ByDescriptionTaggerDataset, tagger_multiclass_collator
from zshot.utils.data_models import Entity
from zshot.utils.data_models import Span


class SmxmInput(dict):
    def __init__(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor,
            sep_index: torch.Tensor,
            seq_mask: torch.Tensor,
            split: torch.Tensor,
            device: torch.device,
    ):
        config = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "token_type_ids": token_type_ids.to(device),
            "sep_index": sep_index.to(device),
            "seq_mask": seq_mask.to(device),
            "split": split.to(device),
        }
        super().__init__(**config)


def predictions_to_span_annotations(
        sentences: List[str],
        predictions: List[List[int]],
        probabilities: List[List[List[float]]],
        entities: List[str],
        tokenizer: BertTokenizerFast,
        max_sentence_tokens: int,
) -> List[List[Span]]:
    span_annotations = []
    for i, sentence_full in enumerate(sentences):
        sentence = sentence_full.split(" ")
        sentence_span_annotations = []
        
        tokenization = tokenizer.encode_plus(
            sentence,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
            is_split_into_words=True 
        )
        
        num_tokens_full_sentence = len(tokenization["input_ids"]) - 2  # Exclude [CLS] and [SEP]
        token_overflow = num_tokens_full_sentence - max_sentence_tokens
        truncation_offset = token_overflow * (token_overflow > 0)
        
        # Get mapping from tokens to words
        mapping_input_id_to_word = tokenization.encodings[0].word_ids
        
        # Create offset mappings for the words in the original sentence
        words_offset_mappings = {}
        word_index_offset = 0
        for word_index, word in enumerate(sentence):
            start_offset = sentence_full.find(word, word_index_offset)
            end_offset = start_offset + len(word)
            words_offset_mappings[word_index] = (start_offset, end_offset)
            word_index_offset = end_offset + 1
        
        # Track entity spans using B-I-O logic similar to the original function
        current_entity = None
        current_start = None
        current_score = 0.0
        
        for j, input_id in enumerate(mapping_input_id_to_word[truncation_offset:-1]):
            if input_id is None:  # Skip special tokens
                continue
                
            if input_id >= len(sentence):  # Handle potential out-of-bounds
                continue
                
            pred = predictions[i][j]
            entity_label = entities[pred]
            
            # Only process first subtoken of each word (like original function's removal of "#" tokens)
            is_first_subtoken = (j == 0) or (input_id != mapping_input_id_to_word[j - 1])
            
            if is_first_subtoken:
                if entity_label != "NEG":
                    # Start new entity or continue current one
                    if current_entity is None:
                        # Start a new entity
                        current_entity = entity_label
                        current_start = words_offset_mappings[input_id][0]
                        current_score = probabilities[i][j][pred]
                    elif current_entity != entity_label:
                        # Different entity - close the current one and start a new one
                        sentence_span_annotations.append(
                            Span(current_start, words_offset_mappings[input_id - 1][1],
                                 current_entity, current_score)
                        )
                        current_entity = entity_label
                        current_start = words_offset_mappings[input_id][0]
                        current_score = probabilities[i][j][pred]
                    else:
                        # Same entity continues - update score if higher
                        current_score = max(current_score, probabilities[i][j][pred])
                else:
                    # End any current entity
                    if current_entity is not None:
                        sentence_span_annotations.append(
                            Span(current_start, words_offset_mappings[input_id - 1][1],
                                 current_entity, current_score)
                        )
                        current_entity = None
            
        # Handle any final entity
        if current_entity is not None and len(mapping_input_id_to_word) > 1:
            last_valid_id = mapping_input_id_to_word[-2]  # Last non-special token
            if last_valid_id is not None and last_valid_id < len(sentence):
                sentence_span_annotations.append(
                    Span(current_start, words_offset_mappings[last_valid_id][1],
                         current_entity, current_score)
                )
                
        span_annotations.append(sentence_span_annotations)
    
    return span_annotations


def get_entities_names_descriptions(
        entities: List[Entity],
) -> Tuple[List[str], List[str]]:
    if "NEG" in [e.name for e in entities]:
        neg_index = [e.name for e in entities].index("NEG")
        entities.insert(0, entities.pop(neg_index))
    else:
        neg_ent = Entity(
            name="NEG",
            description="Coal, water, oil, etc. are normally used for traditional electricity generation. "
                        "However using liquefied natural gas as fuel for joint circulatory electircity generation has advantages. "
                        "The chief financial officer is the only one there taking the fall. It has a very talented team, eh. "
                        "What will happen to the wildlife? I just tell them, you've got to change. They're here to stay. "
                        "They have no insurance on their cars. What else would you like? Whether holding an international cultural event "
                        "or setting the city's cultural policies, she always asks for the participation or input of other cities and counties.",
        )
        entities.insert(0, neg_ent)

    entity_labels = [e.name for e in entities]
    entity_descriptions = [e.description for e in entities]
    entities.pop(0)

    return entity_labels, entity_descriptions


def smxm_predict(model, tokenizer, sentences, entity_labels, entity_descriptions, batch_size):
    encoded_data, max_sentence_tokens = encode_data(
        sentences, entity_labels, entity_descriptions, tokenizer
    )
    dataset = ByDescriptionTaggerDataset(encoded_data)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=partial(tagger_multiclass_collator, device=model.device)
    )

    preds = []
    probabilities = []
    for batch in dataloader:
        with torch.no_grad():
            inputs = SmxmInput(*batch, device=model.device)
            outputs = model(**inputs)
            probability = (
                torch.nn.Softmax(dim=-1)(outputs).cpu().numpy().tolist()
            )
            probabilities += [p for p in probability]
            outputs = torch.argmax(outputs, dim=2)
            preds += outputs.detach().cpu().numpy().tolist()

    span_annotations = predictions_to_span_annotations(
        sentences, preds, probabilities, entity_labels, tokenizer, max_sentence_tokens
    )

    return span_annotations
