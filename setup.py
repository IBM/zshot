from setuptools import setup, find_packages

version = '0.0.1'

setup(name='zshot',
      version=version,
      description="Zero and Few shot named entity recognition",
      long_description="""Zero and Few shot named entity recognition""",
      classifiers=[],
      keywords='NER Zero-Shot Few-Shot',
      author='IBM Research',
      author_email='',
      url='',
      license='Apache 2.0',
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      include_package_data=True,
      zip_safe=False,
      install_requires=[
          "spacy>=3.4.1",
          "requests>=2.28",
          "appdata~=2.1.2",
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
