from zshot.evaluation.dataset import load_few_rel_zs, load_medmentions_zs, load_ontonotes_zs
from zshot.linker import LinkerRegen, LinkerSMXM, LinkerTARS
from zshot.relation_extractor.relation_extractor_zsrc import RelationsExtractorZSRC
from zshot.mentions_extractor.utils import ExtractorType
from zshot.mentions_extractor import MentionsExtractorFlair


def load_all():
    load_few_rel_zs()
    load_medmentions_zs()
    load_ontonotes_zs()
    LinkerSMXM().load_models()
    LinkerRegen().load_models()
    LinkerTARS().load_models()
    RelationsExtractorZSRC().load_models()
    MentionsExtractorFlair(ExtractorType.NER).load_models()
    MentionsExtractorFlair(ExtractorType.POS).load_models()


if __name__ == "__main__":
    load_all()
