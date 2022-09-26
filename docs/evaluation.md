# Evaluation
Evaluation is an important process to keep improving the performance of the models, that's why ZShot allows to evaluate the component with two predefined datasets: OntoNotes and MedMentions, in a Zero-Shot version in which the entities of the test and validation splits don't appear in the train set.  

#### OntoNotes
OntoNotes Release 5.0 is the final release of the OntoNotes project, a collaborative effort between BBN Technologies, the University of Colorado, the University of Pennsylvania and the University of Southern Californias Information Sciences Institute. The goal of the project was to annotate a large corpus comprising various genres of text (news, conversational telephone speech, weblogs, usenet newsgroups, broadcast, talk shows) in three languages (English, Chinese, and Arabic) with structural information (syntax and predicate argument structure) and shallow semantics (word sense linked to an ontology and coreference).

In ZShot the version taken from the [Huggingface datasets](https://huggingface.co/datasets/conll2012_ontonotesv5) is preprocessed to adapt it to the Zero-Shot version. 

[Link](https://catalog.ldc.upenn.edu/LDC2013T19)

#### MedMentions
Corpus: The MedMentions corpus consists of 4,392 papers (Titles and Abstracts) randomly selected from among papers released on PubMed in 2016, that were in the biomedical field, published in the English language, and had both a Title and an Abstract.

Annotators: We recruited a team of professional annotators with rich experience in biomedical content curation to exhaustively annotate all UMLS® (2017AA full version) entity mentions in these papers.

Annotation quality: We did not collect stringent IAA (Inter-annotator agreement) data. To gain insight on the annotation quality of MedMentions, we randomly selected eight papers from the annotated corpus, containing a total of 469 concepts. Two biologists ('Reviewer') who did not participate in the annotation task then each reviewed four papers. The agreement between Reviewers and Annotators, an estimate of the Precision of the annotations, was 97.3%.

In ZShot the data is downloaded from the original repository and preprocessed to convert it into the Zero-Shot version.

[Link](https://github.com/chanzuckerberg/MedMentions)

### How to evaluate ZShot

The package `evaluation` contains all the functionalities to evaluate the ZShot components. The main function is `zshot.evaluation.zshot_evaluate.evaluate`, that will take as input the SpaCy `nlp` model and the dataset(s) and split(s) to evaluate. It will return a `str` containing a table with the results of the evaluation. For instance the evaluation of the ZShot custom component implemented above would be:

```python
import spacy

from zshot import PipelineConfig
from zshot.linker import LinkerRegen
from zshot.mentions_extractor import MentionsExtractorSpacy
from zshot.utils.data_models import Entity
from zshot.evaluation.zshot_evaluate import evaluate
from datasets import Split

nlp = spacy.load("en_core_web_sm")
nlp_config = PipelineConfig(
    mentions_extractor=MentionsExtractorSpacy(),
    linker=LinkerRegen(),
    entities=[
        Entity(name="Paris",
               description="Paris is located in northern central France, in a north-bending arc of the river Seine"),
        Entity(name="IBM",
               description="International Business Machines Corporation (IBM) is an American multinational technology corporation headquartered in Armonk, New York"),
        Entity(name="New York", description="New York is a city in U.S. state"),
        Entity(name="Florida", description="southeasternmost U.S. state"),
        Entity(name="American",
               description="American, something of, from, or related to the United States of America, commonly known as the United States or America"),
        Entity(name="Chemical formula",
               description="In chemistry, a chemical formula is a way of presenting information about the chemical proportions of atoms that constitute a particular chemical compound or molecule"),
        Entity(name="Acetamide",
               description="Acetamide (systematic name: ethanamide) is an organic compound with the formula CH3CONH2. It is the simplest amide derived from acetic acid. It finds some use as a plasticizer and as an industrial solvent."),
        Entity(name="Armonk",
               description="Armonk is a hamlet and census-designated place (CDP) in the town of North Castle, located in Westchester County, New York, United States."),
        Entity(name="Acetic Acid",
               description="Acetic acid, systematically named ethanoic acid, is an acidic, colourless liquid and organic compound with the chemical formula CH3COOH"),
        Entity(name="Industrial solvent",
               description="Acetamide (systematic name: ethanamide) is an organic compound with the formula CH3CONH2. It is the simplest amide derived from acetic acid. It finds some use as a plasticizer and as an industrial solvent."),
    ]
)
nlp.add_pipe("zshot", config=nlp_config, last=True)


evaluation = evaluate(nlp, datasets="ontonotes", 
                      splits=[Split.VALIDATION])
print(evaluation)
```