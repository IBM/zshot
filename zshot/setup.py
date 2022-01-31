from setuptools import setup, find_packages
import sys, os

version = '0.1'

setup(name='zshot',
      version=version,
      description="Zero and Few shot named entity recognition",
      long_description="""\
Zero and Few shot named entity recognition""",
      classifiers=[], # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
      keywords='NER Zero-Shot Few-Shot',
      author='IBM Research',
      author_email='',
      url='',
      license='Apache 2.0',
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      include_package_data=True,
      zip_safe=False,
      install_requires=[
          # -*- Extra requirements: -*-
      ],
      entry_points="""
      # -*- Entry points: -*-
      """,
      )
