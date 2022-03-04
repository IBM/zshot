[![Build Status](https://travis.ibm.com/Dublin-Research-Lab/zshot.svg?token=zSP5krJq4ryG4zqgNyms&branch=master)](https://travis.ibm.com/Dublin-Research-Lab/zshot)
# zshot

Zero and Few shot named entity recognition plugin for Spacy for performing:

- Wikification: the task of linking textual mentions to entities in Wikipedia
- Zero and Few Shot named entity recognition: using language description perform NER to generalize to unseen domains

# Development installation

    pip install -r requirements/devel.txt
    
# Download Spacy transformers model

    python -m spacy download en_core_web_trf 

# Run tests

    python -m pytest -v

# For using Blink NER

    pip install git+https://github.com/facebookresearch/BLINK.git#egg=BLINK


## Examples with Flair, Blink and Displacy

    pip install flair
    pip install git+https://github.com/facebookresearch/BLINK.git#egg=BLINK

Run Wikification on a test sentence

    python -m zshot.examples.wikification
    
![](https://i.imgur.com/0oYuV38.png)
