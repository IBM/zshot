import argparse

import spacy
from zshot import PipelineConfig
from zshot.evaluation import load_medmentions_zs, load_ontonotes_zs
from zshot.evaluation.metrics.seqeval.seqeval import Seqeval
from zshot.evaluation.zshot_evaluate import evaluate, prettify_evaluate_report
from zshot.linker import LinkerTARS, LinkerSMXM, LinkerRegen
from zshot.mentions_extractor import MentionsExtractorSpacy, MentionsExtractorFlair, MentionsExtractorSMXM
from zshot.mentions_extractor.utils import ExtractorType

MENTION_EXTRACTORS = {
    "spacy_pos": lambda: MentionsExtractorSpacy(ExtractorType.POS),
    "spacy_ner": lambda: MentionsExtractorSpacy(ExtractorType.NER),
    "flair_pos": lambda: MentionsExtractorFlair(ExtractorType.POS),
    "flair_ner": lambda: MentionsExtractorFlair(ExtractorType.NER),
    "smxm": lambda: MentionsExtractorSMXM
}
LINKERS = {
    "regen": LinkerRegen,
    "tars": LinkerTARS,
    "smxm": LinkerSMXM
}
END2END = ['tars', 'smxm']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ontonotes", type=str,
                        help="Name or path to the validation data. Comma separated")
    parser.add_argument("--splits", required=False, default="test", type=str,
                        help="Splits to evaluate. Comma separated")
    parser.add_argument("--mode", required=False, default="full", type=str,
                        help="Evaluation mode. One of: full; mentions_extractor; linker")
    parser.add_argument("--mentions_extractor", required=False, default="all", type=str,
                        help="Mentions extractor to evaluate. One of: all; spacy_pos; spacy_ner; flair_pos; flair_ner")
    parser.add_argument("--linker", required=False, default="all", type=str,
                        help="Linker to evaluate. One of: all; tars")
    args = parser.parse_args()

    splits = args.splits.split(",")
    datasets = args.dataset.split(",")

    configs = {}
    if args.mentions_extractor == "all":
        mentions_extractors = list(MENTION_EXTRACTORS.keys())
    else:
        mentions_extractors = [args.mentions_extractor]

    if args.linker == "all":
        linkers = list(LINKERS.keys())
    else:
        linkers = [args.linker]

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
                nlp.get_pipe("zshot").mentions = dataset.entities
                nlp.get_pipe("zshot").entities = dataset.entities
                field_names = ["Metric"]
                field_name = f"{dataset_name} {split}"
                field_names.append(field_name)

                evaluation = evaluate(nlp, dataset, metric=Seqeval())
                print(prettify_evaluate_report(evaluation, name=f"{dataset_name}-{split}"))
