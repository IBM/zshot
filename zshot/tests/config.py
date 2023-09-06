from zshot.utils.data_models import Entity, Relation

EX_DOCS = ["The Domain Name System (DNS) is the hierarchical and decentralized naming system used to identify"
           " computers, services, and other resources reachable through the Internet or other Internet Protocol"
           " (IP) networks.",
           "International Business Machines Corporation (IBM) is an American multinational technology corporation"
           " headquartered in Armonk, New York, with operations in over 171 countries."]

EX_ENTITIES = \
    [
        Entity(name="apple", description="the apple fruit"),
        Entity(name="DNS", description="domain name system", vocabulary=["DNS", "Domain Name System"]),
        Entity(name="IBM", description="technology corporation", vocabulary=["IBM", "International Business machine"]),
        Entity(name="NYC", description="New York city"),
        Entity(name="Florida", description="southeasternmost U.S. state"),
        Entity(name="Paris", description="Paris is located in northern central France, "
                                         "in a north-bending arc of the river Seine"),
    ]

EX_RELATIONS = \
    [
        Relation(name="parent", description="Is the parent of someone"),
        Relation(name="child", description="Is the child of someone"),
        Relation(name="sibling", description="Is the sibling of someone"),
        Relation(name='crosses',
                 description='obstacle (body of water, road, ...) '
                             'which this bridge crosses over or this tunnel goes under'),
    ]

EX_DATASET_RELATIONS = {
    'sentences': [
        'In June 1987 , the Missouri Highway and Transportation Department approved design location of a '
        'new four - lane Mississippi River bridge to replace the deteriorating Cape Girardeau Bridge .',
        'Wilton Bridge   was a major crossing   of the River Wye and was protected by Wilton Castle .'],
    'sentence_entities': [[{'end': 187,
                            'label': 'Q5034838',
                            'sentence': 'In June 1987 , the Missouri Highway and Transportation Department '
                                        'approved design location of a new four - lane '
                                        'Mississippi River bridge to replace the deteriorating Cape Girardeau Bridge .',
                            'start': 165},
                           {'end': 129,
                            'label': 'Q1497',
                            'sentence': 'In June 1987 , the Missouri Highway and Transportation Department '
                                        'approved design location of a new four - lane Mississippi River '
                                        'bridge to replace the deteriorating Cape Girardeau Bridge .',
                            'start': 111}],
                          [{'end': 13,
                            'label': 'Q8023362',
                            'sentence': 'Wilton Bridge   was a major crossing   of the River Wye '
                                        'and was protected by Wilton Castle .',
                            'start': 0},
                           {'end': 55,
                            'label': 'Q19695',
                            'sentence': 'Wilton Bridge   was a major crossing   of the River Wye '
                                        'and was protected by Wilton Castle .',
                            'start': 45}]],
    'labels': ['crosses', 'crosses']}

TEXT_CUBESAT = "The Italian Space Agency’s Light Italian CubeSat for Imaging of Asteroids, or LICIACube, will fly by Dimorphos to capture images and video of the impact plume as it sprays up off the asteroid and maybe even spy the crater it could leave behind."
TEXT_ACETAMIDE = "CH2O2 is a chemical compound similar to Acetamide used in International Business Machines Corporation (IBM)."
TEXTS = [TEXT_CUBESAT, TEXT_ACETAMIDE]
