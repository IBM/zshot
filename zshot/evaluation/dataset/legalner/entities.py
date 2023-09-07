from zshot.utils.data_models import Entity


LEGALNER_SPLITS = {
    "train": ['COURT', 'PETITIONER', 'RESPONDENT', 'LAWYER', "PROVISION",
                    'OTHER_PERSON'],
    "validation": ['JUDGE', 'ORG', 'STATUTE'],
    "test": ['GPE', 'PRECEDENT', 'CASE_NUMBER', 'WITNESS']
}


LEGALNER_ENTITIES = [
    Entity(name='COURT',
           description='The name of the court that has delivered the current judgment if extracted from the preamble. It also includes the name of any court mentioned if extracted from judgment sentences.\n',
           vocabulary=None),
    Entity(name='PETITIONER',
           description='The name of the petitioners, appellants, or revisionists from the current case.\n',
           vocabulary=None),
    Entity(name='RESPONDENT',
           description='The name of the respondents, defendants, or opposition from the current case.\n',
           vocabulary=None),
    Entity(name='LAWYER',
           description='The name of the lawyers from both parties mentioned in the preamble.\n',
           vocabulary=None),
    Entity(name='PROVISION',
           description='Sections, sub-sections, articles, orders, rules under a statute mentioned in the judgment.\n',
           vocabulary=None),
    Entity(name='OTHER_PERSON',
           description='The names of all the people not included in petitioner, respondent, judge, and witness categories mentioned in the judgment.\n',
           vocabulary=None),
    Entity(name='JUDGE',
           description='The name of the judges from the current case if extracted from the preamble. It also includes the name of the judges of the current or previous cases if extracted from judgment sentences.\n',
           vocabulary=None),
    Entity(name='ORG',
           description='The names of organizations mentioned in the text apart from the court. For example, banks, public sector undertakings (PSUs), private companies, police stations, state governments, etc.\n',
           vocabulary=None),
    Entity(name='STATUTE', description='The name of the act or law mentioned in the judgment.\n',
           vocabulary=None),
    Entity(name='GPE',
           description='Geopolitical locations that include names of countries, states, cities, districts, or villages mentioned in the judgment.\n',
           vocabulary=None),
    Entity(name='PRECEDENT',
           description='All the past court cases referred to in the judgment as precedent. The precedent consists of party names, citation (optional), or case number (optional).\n',
           vocabulary=None),
    Entity(name='CASE_NUMBER',
           description='All the other case numbers mentioned in the judgment (apart from precedent) where party names and citation are not provided.\n',
           vocabulary=None),
    Entity(name='WITNESS', description='The name of witnesses mentioned in the current judgment.\n',
           vocabulary=None)
]
