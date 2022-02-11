[![Build Status](https://travis.ibm.com/Dublin-Research-Lab/zshot.svg?token=zSP5krJq4ryG4zqgNyms&branch=master)](https://travis.ibm.com/Dublin-Research-Lab/zshot)
# zshot

Zero and Few shot named entity recognition plugin for Spacy

# Development installation

    pip install -r requirements/devel.txt

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
