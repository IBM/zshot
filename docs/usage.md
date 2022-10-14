# Usage

# Introduction

This is an introductory notebook to walk through all the functionalities of the **ZShot** plugin for Spacy.


```python
from typing import Iterable

import spacy
from spacy.tokens import Doc
from spacy import displacy

from zshot.utils.data_models import Entity, Span
from zshot.mentions_extractor import MentionsExtractor, MentionsExtractorSpacy, MentionsExtractorFlair
from zshot.mentions_extractor.utils import ExtractorType
from zshot.linker import Linker, LinkerSMXM, LinkerTARS, LinkerBlink
from zshot import PipelineConfig

from datasets import Split
```

### Create the Spacy model with the  ZShot component 

The first thing is to create the Spacy Model (a.k.a. `nlp`) with the ZShot component to perform **zero-shot NERC**.

In order to do that, a `nlp` model has to be created first. Depending on the language, there are several models available:

 - `blank`: Spacy blank model. It has no trained pipelines.
 - `sm`: Spacy small model. It is faster than other but use to have less vocabulary and worse performance.
 - `md`: Spacy medium model. Slower than small models but with more vocabulary and better performance.
 - `lg`: Spacy large model. Slower than medium models but with more vocabulary and better performance.
 - `trf`: Spacy model based on Transformers. Slower that large models but with better performance.
 
*Note: For most of these models there will be also several options available, depending on the source of the data they were trained on*

Wich one should you use? Well, it depends on the mentions extractor and the linker you are going to use. If you rare going to use some Spacy-based mentions extractor you can't use the `blank` model. For the rest of mentions extractors and linker you can select the one that suits you the best.

In this example the model based on transformers is going to be used.


```python
nlp = spacy.load("en_core_web_trf")
```

There are three main steps in order to create the ZShot component:
 1. Add the `entities` to be extracted. They are **zero-shot** models not wizards, they don't know what you want.
 2. Select the `mentions extractor` to use. The `mentions extractor` will extract broad mentions without a specific `entity` assigned.
 3. Select the `linker` to use. The `linker` will link the mentions extracted by the `mentions extractor` and will link them to a specific `entity`. Some of the `linkers` are `end2end`, this is, they don't need a `mentions extractor` and therefore this field can be left empty.

#### Select Entities 

In order to specify the *entities* you can use the `Entity` class provided, that will have a label and a description. This description may be used by some linkers to improve the performance.


```python
entities = [
    Entity(name="company", description="The name of a company"),
    Entity(name="location", description="A physical location"),
    Entity(name="chemical compound", description="Any of a large class of chemical compounds in " \
           "which one or more atoms of carbon are covalently linked to atoms of other elements, " \
           "most commonly hydrogen, oxygen, or nitrogen")
], 
```

You can also use python `dict` (e.g.: loaded from a JSON).


```python
entities = [
    {
        'name': "company", 
        'description': "The name of a company"
    },
    {
        'name': "location", 
        'description': "A physical location"
    },
    {
        'name': "chemical compound", 
        'description': " Any of a large class of chemical compounds in " \
           "which one or more atoms of carbon are covalently linked to atoms of other elements, " \
           "most commonly hydrogen, oxygen, or nitrogen"
    }
], 
```

Or, if the `linker` you're going to use doesn't require the descriptions, you can use a `list` of strings containing the labels.


```python
entities = [
    "company",
    "location",
    "chemical compound"
]
```

#### Select Mention Extractor 

The `mentions_extractor` is the component that will extract broad mentions without a specific `entity` assigned. 

Currently, 2 different `mentions_extractor` are provided:

 - `MentionsExtractorSpacy`
 - `MentionsExtractorFlair`
  
To create a `mentions_extractor` just instantiate the class with the version to be used. There are two different versions for each `mentions_extractor`:

  - NER-Based: Will use a NER model to extract the mentions. 
  - POS-Based: Will use PoS tagging to extract the mentions.
  
You can obtain them from the `ExtractorType`.


```python
mentions_extractor = MentionsExtractorSpacy(ExtractorType.NER)
```

#### Select Linker

The `linker` is the component that will link the extracted mentions to a specific `entity`. Some of them are `end2end`, this is, they don't need and won't use the `mentions_extractor`. 

Currently, 4 different `linker` are provided:

 - `LinkerBLINK`: See [this](https://github.com/facebookresearch/BLINK)
 - `LinkerRegen` See [this](https://github.com/facebookresearch/GENRE)
 - `LinkerSMXM`: `end2end` model that uses *descriptions*. See [this](https://github.com/Raldir/Zero-shot-NERC)
 - `LinkerTARS`: `end2end` model. See [this](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_10_TRAINING_ZERO_SHOT_MODEL.md)


```python
linker = LinkerTARS()
```

#### Create the Pipeline Config 

Once that the `entities`, the `mentions_extractor` and the `linker` are selected, the `PipelineConfig` can be created to configure the **ZShot** component.


```python
config = PipelineConfig(
    entities=entities,
    mentions_extractor=mentions_extractor,
    linker=linker
)
```

Or you can create everythin on the fly:


```python
config = PipelineConfig(
    entities=[
        Entity(name="company", description="The name of a company"),
        Entity(name="location", description="A physical location"),
        Entity(name="chemical compound", description="Any of a large class of chemical compounds in which one or more atoms of carbon are covalently linked to atoms of other elements, most commonly hydrogen, oxygen, or nitrogen")
    ], 
    linker=LinkerSMXM()
)
```

#### Create the component 

Once the `PipelineConfg` has been created it's time to create the **ZShot** component and add it to the `nlp` pipe. Use the `last=True` option to assure the model is added to the end of the pipe, as some components have to be executed first.


```python
nlp.add_pipe("zshot", config=config, last=True)
```
    WARNING:root:Disabling default NER
    <zshot.zshot.Zshot at 0x12cc4fa60>

### Execute 

Now you can use the `nlp` model as always to see the entities extracted!


```python
text_acetamide = "CH2O2 is a chemical compound similar to Acetamide used in International Business " \
        "Machines Corporation (IBM) to create new materials that act like PAGs."

doc = nlp(text_acetamide)
displacy.render(doc, style="ent")
```

<span class="tex2jax_ignore"><div class="entities" style="line-height: 2.5; direction: ltr">
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    CH2O2
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">chemical compound</span>
</mark>
 is a chemical compound similar to 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Acetamide
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">chemical compound</span>
</mark>
 used in 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    International Business Machines Corporation
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">company</span>
</mark>
 (
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    IBM
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">company</span>
</mark>
) to create new materials that act like PAGs.</div></span>



```python
for ent in doc.ents:
    print(ent.text, "-", ent.label_)
```

    CH2O2 - chemical compound
    Acetamide - chemical compound
    International Business Machines Corporation - company
    IBM - company


##### Use our display

If you don't like the gray color of displacy, or you want different colors for each entity, you can use our displacy tool


```python
from zshot.utils.displacy import displacy
```


```python
text_acetamide = "CH2O2 is a chemical compound similar to Acetamide used in International Business " \
        "Machines Corporation (IBM) to create new materials that act like PAGs."

doc = nlp(text_acetamide)
displacy.render(doc, style="ent")
```


<span class="tex2jax_ignore"><div class="entities" style="line-height: 2.5; direction: ltr">
<mark class="entity" style="background: #072beb; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    CH2O2
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">chemical compound</span>
</mark>
 is a chemical compound similar to 
<mark class="entity" style="background: #072beb; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Acetamide
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">chemical compound</span>
</mark>
 used in 
<mark class="entity" style="background: #e2f779; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    International Business Machines Corporation
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">company</span>
</mark>
 (
<mark class="entity" style="background: #e2f779; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    IBM
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">company</span>
</mark>
) to create new materials that act like PAGs.</div></span>


### Use your own component 

If you want to implement your own `mentions_extractor` or `linker` and use it with **ZShot** you can do it. To make it easier for the user to implement a new component, some base classes are provided that you have to extend with your code.

It is as simple as create a new class extending the base class (`MentionsExtractor` or `Linker`). You will have to implement the predict method, which will receive the *Spacy Documents* and will return a list of `zshot.utils.data_models.Span` for each document.

Let's create a simple `mentions_extractor` that will extract as mentions all words that contain the letter *s*:


```python
class SimpleMentionExtractor(MentionsExtractor):
    def predict(self, docs: Iterable[Doc], batch_size=None):
        spans = [[Span(tok.idx, tok.idx + len(tok)) for tok in doc if "s" in tok.text] for doc in docs]
        return spans
```

Now, let's create a new `nlp` model with a **ZShot** component with the new `mentions_extractor`


```python
new_nlp = spacy.load("en_core_web_trf")

config = PipelineConfig(
    mentions_extractor=SimpleMentionExtractor()
)
new_nlp.add_pipe("zshot", config=config, last=True)
```
    WARNING:root:Disabling default NER
    <zshot.zshot.Zshot at 0x12df01e10>

And let's try it:


```python
text_acetamide = "CH2O2 is a chemical compound similar to Acetamide used in International Business " \
        "Machines Corporation (IBM) to create new materials that act like PAGs."

doc = new_nlp(text_acetamide)
print(doc._.mentions)
```

    [is, similar, used, Business, Machines, materials, PAGs]


### Evaluation 

If you have a new **ZShot** component maybe you want to evaluate it over some famous benchmarks to get an idea of the performance of your model.

**ZShot** evaluation package contains all you need to do it. It makes it easy for the user to evaluate the component over a Zero-Shot dataset.

The list of the datasets available at the moment is:
 - OntoNotes. See [this](https://catalog.ldc.upenn.edu/LDC2013T19)
 - MedMentions. See [this](https://github.com/chanzuckerberg/MedMentions/)

Now you can use the `evaluate` function to evaluate your `nlp` over a dataset.

You can evaluate one or more dataset, and using just one or more splits.

```python
def evaluate(nlp: spacy.Language,
             datasets: Union[str, List[str]],
             splits: Optional[Union[str, List[str]]] = None) -> str:
    """ Evaluate a spacy zshot model

    :param nlp: Spacy Language pipeline with ZShot components
    :param datasets: Dataset or list of datasets to evaluate
    :param splits: Optional. Split or list of splits to evaluate. All splits available by default
    :return: Result of the evaluation. String containing a table with the result
    """
```


```python
from zshot.evaluation.zshot_evaluate import evaluate
from datasets import Split

evaluation = evaluate(new_nlp, "ontonotes", 
                      splits=[Split.VALIDATION])
print(evaluation)
```