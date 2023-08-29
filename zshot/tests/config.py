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
    ]


text_cubesat = "The Italian Space Agency’s Light Italian CubeSat for Imaging of Asteroids, or LICIACube, will fly by Dimorphos to capture images and video of the impact plume as it sprays up off the asteroid and maybe even spy the crater it could leave behind."
text_acetamide = "CH2O2 is a chemical compound similar to Acetamide used in International Business Machines Corporation (IBM)."
texts = [text_cubesat, text_acetamide]
