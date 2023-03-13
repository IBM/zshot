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

The package `evaluation` contains all the functionalities to evaluate the ZShot components. The main function is `zshot.evaluation.zshot_evaluate.evaluate`, that will take as input the SpaCy `nlp` model and the dataset to evaluate. It will return a `str` containing a table with the results of the evaluation. For instance the evaluation of the TARS linker in ZShot for the *Ontonotes validation* set would be:

```python
import spacy

from zshot import PipelineConfig
from zshot.linker import LinkerTARS
from zshot.evaluation.dataset import load_ontonotes_zs
from zshot.evaluation.zshot_evaluate import evaluate, prettify_evaluate_report
from zshot.evaluation.metrics.seqeval.seqeval import Seqeval

ontonotes_zs = load_ontonotes_zs('validation')


nlp = spacy.blank("en")
nlp_config = PipelineConfig(
    linker=LinkerTARS(),
    entities=ontonotes_zs.entities
)

nlp.add_pipe("zshot", config=nlp_config, last=True)

evaluation = evaluate(nlp, ontonotes_zs, metric=Seqeval())
prettify_evaluate_report(evaluation)
```
