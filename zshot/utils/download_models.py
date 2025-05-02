from zshot.linker import LinkerRegen, LinkerSMXM, LinkerTARS, LinkerGLINER, LinkerRelik
from zshot.mentions_extractor import MentionsExtractorFlair
from zshot.mentions_extractor.utils import ExtractorType
from zshot.relation_extractor.relation_extractor_zsrc import RelationsExtractorZSRC


def load_all():
    try:
        LinkerSMXM().load_models()
    except RuntimeError:
        pass
    try:
        LinkerRegen().load_models()
    except RuntimeError:
        pass
    try:
        LinkerTARS().load_models()
    except RuntimeError:
        pass
    try:
        LinkerGLINER().load_models()
    except RuntimeError:
        pass
    # try:
    #     LinkerRelik().load_models()
    # except RuntimeError:
    #     pass
    try:
        RelationsExtractorZSRC().load_models()
    except RuntimeError:
        pass
    try:
        MentionsExtractorFlair(ExtractorType.NER).load_models()
    except RuntimeError:
        pass
    try:
        MentionsExtractorFlair(ExtractorType.POS).load_models()
    except RuntimeError:
        pass


if __name__ == "__main__":
    load_all()
