language: python
python:
  - "3.9"
# command to install dependencies
install:
  - pip install -U pip
  - pip install -r requirements/devel.txt
# command to run tests
script:
  - python -m pytest -v