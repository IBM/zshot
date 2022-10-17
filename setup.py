from setuptools import setup, find_packages
from pathlib import Path

from zshot import name, version, author, url, license_

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(name=name,
      version=version,
      description="Zero and Few shot named entity recognition",
      long_description_content_type='text/markdown',
      long_description=long_description,
      classifiers=[],
      keywords='NER Zero-Shot Few-Shot',
      author=author,
      author_email='',
      url=url,
      license=license_,
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      include_package_data=True,
      zip_safe=False,
      install_requires=[
          "spacy>=3.4.1",
          "requests>=2.28",
          "tqdm>=4.62.3",
          "setuptools~=60.0.0",  # Needed to install dynamic packages from source (e.g. Blink)
          "prettytable>=3.4",
          "torch>=1",
          "transformers>=4.20",
          "datasets>=2.2.2",
          "evaluate>=0.2.2",
          "seqeval>=1.2.2",
      ],
      entry_points="""
      # -*- Entry points: -*-
      """,
      )
