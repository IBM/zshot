# Zshot

<p align="center">
  <a href="https://fastapi.tiangolo.com"><img src="./img/zshot-header.png"></a>
</p>
<p align="center">
    <em>Zero and Few shot Named Entities and Relationships recognition</em>
</p>
<p align="center">
<a href="https://travis.ibm.com/Dublin-Research-Lab/zshot" target="_blank">
    <img src="https://travis.ibm.com/Dublin-Research-Lab/zshot.svg?token=zSP5krJq4ryG4zqgNyms&branch=master" alt="Test">
</a>
</p>

---

**Documentation**: <a href="https://pages.github.ibm.com/Dublin-Research-Lab/zshot" target="_blank">https://pages.github.ibm.com/Dublin-Research-Lab/zshot</a>

**Source Code**: <a href="https://github.ibm.com/Dublin-Research-Lab/zshot" target="_blank">https://github.ibm.com/Dublin-Research-Lab/zshot</a>

---

Zshot is a highly customasible framework for performing Zero and Few shot named entity recognition.

Can be used to perform:

- **Mentions extraction**: Identify globally relevant mentions or mentions relevant for a given domain 
- **Wikification**: The task of linking textual mentions to entities in Wikipedia
- **Zero and Few Shot named entity recognition**: using language description perform NER to generalize to unseen domains (work in progress)
- **Zero and Few Shot named relationship recognition** (work in progress)

## Requirements

Python 3.6+

Zshot stands on the shoulders of giants:

* <a href="https://spacy.io/" class="external-link" target="_blank">Spacy</a>.

## Installation

<div class="termy">

```console
$ pip install -r requirements.txt

---> 100%
```

</div>

You will also need a Spacy transformers model

<div class="termy">

```console
$ python -m spacy download en_core_web_trf 

---> 100%
```

</div>

## Example

### Install addiotional dependencies

Install [Flair](https://github.com/facebookresearch/BLINK/tree/main/blink) to use the flair for the mentions extraction

```console
$ pip install flair==0.10

---> 100%
```

Install [Blink](https://github.com/facebookresearch/BLINK/tree/main/blink) to use the Blink linker

```console
$ pip install git+https://github.com/facebookresearch/BLINK.git#egg=BLINK

---> 100%
```

### Example of use

* Create a file `main.py` with:

```Python
import spacy
from spacy import displacy

from zshot import Linker, MentionsExtractor

text = "International Business Machines Corporation (IBM) is an American multinational technology corporation " \
       "headquartered in Armonk, New York, with operations in over 171 countries."

nlp = spacy.load("en_core_web_trf")
nlp.disable_pipes('ner')
nlp.add_pipe("zshot", config={"mentions_extractor": MentionsExtractor.FLAIR, "linker": Linker.BLINK}, last=True)
print(nlp.pipe_names)

doc = nlp(text)
displacy.serve(doc, style="ent")
```


<details markdown="1">
<summary>Or run the prepared example</summary>
```console
$ python -m zshot.examples.wikification
```
</details>


### Run it

Run with

```console
$ python main.py

Using the 'ent' visualizer
Serving on http://0.0.0.0:5000 ...
```


The script will annotate the text using Zshot and use Displacy for visualising the annotations

### Check it

Open your browser at <a href="http://127.0.0.1:5000" class="external-link" target="_blank">http://127.0.0.1:5000</a>.

You will see the annotated sentence:

<img src="./img/annotations.png">

## Optional Dependencies

Mentions extraction:

* <a href="https://github.com/flairNLP/flair" target="_blank"><code>Flair</code></a> - Required if you want to use Flair mentions extractor.
Entity linking:

* <a href="https://github.com/facebookresearch/BLINK" target="_blank"><code>Blink</code></a> - Required if you want to use Blink for linking to Wikipedia pages.