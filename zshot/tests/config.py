from zshot.entity import Entity

EX_DOCS = ["The Domain Name System (DNS) is the hierarchical and decentralized naming system used to identify"
           " computers, services, and other resources reachable through the Internet or other Internet Protocol"
           " (IP) networks.",
           "International Business Machines Corporation (IBM) is an American multinational technology corporation"
           " headquartered in Armonk, New York, with operations in over 171 countries."]

EX_ENTITIES_DICT = {"apple": "The apple fruit",
                    "DNS": "domain name system",
                    "IBM": "technology corporation",
                    "NYC": "New York city",
                    "Florida": "southeasternmost U.S. state",
                    "Paris": "Paris is located in northern central France, "
                             "in a north-bending arc of the river Seine"}

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

EX_ENTITIES_NEG = [
    Entity(name="NEG", description="Coal, water, oil, etc. are normally used for traditional electricity generation. "
                                   "However using liquefied natural gas as fuel for joint circulatory electircity generation has advantages. "
                                   "The chief financial officer is the only one there taking the fall. It has a very talented team, eh. "
                                   "What will happen to the wildlife? I just tell them, you've got to change. They're here to stay. "
                                   "They have no insurance on their cars. What else would you like? Whether holding an international cultural event "
                                   "or setting the city's cultural policies, she always asks for the participation or input of other cities and counties."),
]
EX_ENTITIES_NEG.extend(EX_ENTITIES)
