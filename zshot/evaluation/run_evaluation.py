import argparse

import spacy

from zshot import PipelineConfig
from zshot.evaluation import load_medmentions_zs, load_ontonotes_zs
from zshot.evaluation.dataset.dataset import DatasetWithEntities
from zshot.evaluation.dataset.med_mentions.entities import MEDMENTIONS_EXPLANATORY_MAPPING, \
    MEDMENTIONS_EXPLANATORY_INVERSE_MAPPING
from zshot.evaluation.dataset.ontonotes.entities import ONTONOTES_EXPLANATORY_INVERSE_MAPPING, \
    ONTONOTES_EXPLANATORY_MAPPING
from zshot.evaluation.metrics._seqeval._seqeval import Seqeval
from zshot.evaluation.zshot_evaluate import evaluate, prettify_evaluate_report
from zshot.linker import LinkerTARS, LinkerSMXM, LinkerRegen, LinkerGLINER
from zshot.mentions_extractor import MentionsExtractorSpacy, MentionsExtractorFlair, \
    MentionsExtractorSMXM, MentionsExtractorTARS, MentionsExtractorGLINER
from zshot.mentions_extractor.utils import ExtractorType

MENTION_EXTRACTORS = {
    "spacy_pos": lambda: MentionsExtractorSpacy(ExtractorType.POS),
    "spacy_ner": lambda: MentionsExtractorSpacy(ExtractorType.NER),
    "flair_pos": lambda: MentionsExtractorFlair(ExtractorType.POS),
    "flair_ner": lambda: MentionsExtractorFlair(ExtractorType.NER),
    "smxm": lambda: MentionsExtractorSMXM,
    "tars": lambda: MentionsExtractorTARS,
    "gliner": lambda: MentionsExtractorGLINER
}
LINKERS = {
    "regen": LinkerRegen,
    "tars": LinkerTARS,
    "smxm": LinkerSMXM,
    "gliner": LinkerGLINER
}
END2END = ['tars', 'smxm', 'gliner']
ENTITIES_MAPPERS = {
    "medmentions": MEDMENTIONS_EXPLANATORY_MAPPING,
    "ontonotes": ONTONOTES_EXPLANATORY_MAPPING
}
ENTITIES_INVERSE_MAPPERS = {
    "medmentions": MEDMENTIONS_EXPLANATORY_INVERSE_MAPPING,
    "ontonotes": ONTONOTES_EXPLANATORY_INVERSE_MAPPING
}


def convert_entities(dataset: DatasetWithEntities, dataset_name: str) -> DatasetWithEntities:
    mapping = ENTITIES_MAPPERS[dataset_name]
    for entity in dataset.entities:
        entity.name = mapping[entity.name]

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ontonotes", type=str,
                        help="Name or names of the datasets. One of: ontonotes; medmentions. Comma separated")
    parser.add_argument("--splits", required=False, default="test", type=str,
                        help="Splits to evaluate. Comma separated")
    parser.add_argument("--mode", required=False, default="full", type=str,
                        help="Evaluation mode. One of: full; mentions_extractor; linker")
    parser.add_argument("--entity_name", default="original", type=str,
                        help="Type of entity name. One of: original; explanatory. Original by default.")
    parser.add_argument("--mentions_extractor", required=False, default="all", type=str,
                        help="Mentions extractor to evaluate. "
                             "One of: all; spacy_pos; spacy_ner; flair_pos; flair_ner; smxm; tars; gliner")
    parser.add_argument("--linker", required=False, default="all", type=str,
                        help="Linker to evaluate. One of: all; regen; smxm; tars; gliner")
    parser.add_argument("--show_full_report", action="store_false",
                        help="Show evalution report for each label. True by default")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--span_evaluation", action="store_true", help="Evaluate based on Spans")
    group.add_argument("--token_evaluation", action="store_true", help="Evaluate based on Token")

    args = parser.parse_args()

    splits = args.splits.split(",")
    datasets = args.dataset.split(",")

    mode = 'token' if args.token_evaluation else 'span'

    configs = {}
    if args.mentions_extractor == "all":
        mentions_extractors = list(MENTION_EXTRACTORS.keys())
    else:
        mentions_extractors = [args.mentions_extractor]

    if args.linker == "all":
        linkers = list(LINKERS.keys())
    else:
        linkers = args.linker.split(',')

    if args.mode == "full":
        if linkers:
            for linker in linkers:
                if linker not in END2END:
                    for mentions_extractor in mentions_extractors:
                        configs[f"{mentions_extractor}_{linker}"] = PipelineConfig(
                            mentions_extractor=MENTION_EXTRACTORS[mentions_extractor](),
                            linker=LINKERS[linker]()
                        )
                else:
                    configs[linker] = PipelineConfig(linker=LINKERS[linker]())
    elif args.mode == "mentions_extractor":
        for mentions_extractor in mentions_extractors:
            configs[mentions_extractor] = PipelineConfig(mentions_extractor=MENTION_EXTRACTORS[mentions_extractor]())
    elif args.mode == "linker":
        for linker in linkers:
            if linker not in END2END:
                configs[linker] = PipelineConfig(mentions_extractor=MENTION_EXTRACTORS['spacy_ner'](),
                                                 linker=LINKERS[linker]())
            else:
                configs[linker] = PipelineConfig(linker=LINKERS[linker]())

    print(f"Evaluating {args.dataset} with splits {args.splits}")
    for key, config in configs.items():
        print(f"Evaluating {key}")
        nlp = spacy.blank("en") if "spacy" not in key else spacy.load("en_core_web_sm")
        nlp.add_pipe("zshot", config=config, last=True)

        for dataset_name in datasets:
            for split in splits:
                if dataset_name.lower() == "medmentions":
                    dataset = load_medmentions_zs(split)
                elif dataset_name.lower() == "ontonotes":
                    dataset = load_ontonotes_zs(split)
                else:
                    raise ValueError(f"{dataset_name} not supported")

                if args.entity_name == "explanatory":
                    convert_entities(dataset, dataset_name)

                nlp.get_pipe("zshot").mentions = dataset.entities
                nlp.get_pipe("zshot").entities = dataset.entities

                evaluation = evaluate(nlp, dataset, metric=Seqeval(), mode=mode,
                                      entity_mapper=ENTITIES_MAPPERS[dataset_name] if args.entity_name != "original"
                                      else None)
                print("\n".join(prettify_evaluate_report(evaluation, name=f"{dataset_name}-{split}")))
