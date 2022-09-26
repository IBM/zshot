from zshot.utils.data_models import Entity

ONTONOTES_ENTITIES = [Entity(name='NEG',
                             description="Coal, water, oil, etc. are normally used for traditional electricity "
                                         "generation. However using liquefied natural gas as fuel for joint "
                                         "circulatory electircity generation has advantages. The chief financial "
                                         "officer is the only one there taking the fall. It has a very talented team, "
                                         "eh. What will happen to the wildlife? I just tell them, you've got to change."
                                         " They're here to stay. They have no insurance on their cars. "
                                         "What else would you like? Whether holding an international cultural event "
                                         "or setting the city's cultural policies, she always asks for the "
                                         "participation or input of other cities and counties.",
                             vocabulary=None),
                      Entity(name='PERSON',
                             description='These are proper names of people, including fictional people, first names, '
                                         'last names, individual or family names, unique nicknames. '
                                         'Generational markers (Jr., IV) are included in the extent, while honorifics '
                                         '(Ms., Dr.) and occupational titles (President, Secretary) are NOT included.',
                             vocabulary=None),
                      Entity(name='NORP',
                             description='This type represents adjectival forms of GPE and Location names, as well '
                                         'as adjectival forms of named religions, heritage and political affiliation. '
                                         'Also marked are head words which refer to people using the name of an entity '
                                         'with which they are affiliated, often a GPE or Organization. The distinction '
                                         'between NORP and other types is morphological. "American" and "Americans" '
                                         'are adjectival nationalities, while "America" and "US" are GPEs, '
                                         'regardless of context.',
                             vocabulary=None),
                      Entity(name='FAC',
                             description='Names of man-made structures: infrastructure (streets, bridges), buildings, '
                                         'monuments, etc. belong to this type. Buildings that are referred to using '
                                         'the name of the company or organization that uses them should be marked as '
                                         'FAC when they refer to the physical structure of the building itself, '
                                         'usually in a locative way: "I\'m reporting live from right outside '
                                         '[Massachusetts General Hospital]"',
                             vocabulary=None),
                      Entity(name='ORG',
                             description='These are names of companies, government agencies, educational institutions, '
                                         'sport teams. Names of hospitals, museums, and libraries should be marked, '
                                         'unless the mentions are clearly referring to the building in a locative way.'
                                         ' Adjectival forms of organization names are included, as are metonymic '
                                         'mentions of the buildings or locations associated with these organizations.'
                                         ' A group, team, force, etc. must be officially sanctioned in some way to be '
                                         'classified as an Organization. Organized crime groups, such as the Mafia, '
                                         'are not marked. Terrorist groups such as Al-Qaeda,'
                                         ' however, should be marked.',
                             vocabulary=None),
                      Entity(name='GPE',
                             description='Names of countries, cities, states, provinces, municipalities. '
                                         'In cases where the GPE name is modified, such as "southern California," '
                                         '[California] is marked as a GPE name and there is NO other markup.',
                             vocabulary=None),
                      Entity(name='LOC',
                             description='Names of geographical locations other than GPEs. These include '
                                         'mountain ranges, coasts, borders, planets, geo-coordinates, '
                                         'bodies of water. Also included in this category are named regions such as '
                                         'the Middle East, areas, neighborhoods, continents and regions of continents.'
                                         ' Do NOT mark deictics or other non-proper nouns: here, there, everywhere,'
                                         ' etc. As with GPEs, directional modifiers such as "southern" are only '
                                         'marked when they are part of the location name itself.',
                             vocabulary=None),
                      Entity(name='PRODUCT',
                             description='This can be name of any product, generally a model name or model '
                                         'name and number. Named foods are also included. Credit cards, checking'
                                         ' accounts, CDs, and credit plans are NOT marked. References that include'
                                         ' manufacturer and product should be marked as two separate named entities,'
                                         ' ORG + PRODUCT: [Apple] [iPod], [Dell] [Inspiron], [Ford] [Mustang].',
                             vocabulary=None),
                      Entity(name='DATE',
                             description='Used to classify a reference to a date or period, etc. Age also falls '
                                         'under this category, even when it’s a noun phrase referring to a person: '
                                         'the 5-year-old, 5 years old, Jane Doe, 5, etc. Extent should include '
                                         'modifiers & prepositions that denote specific time, such as [2 days ago], '
                                         '[the past two days], but not those that mark duration, such as "for '
                                         '[2 days]." Do not separate mentions into their component parts: [November'
                                         ' 2, 2001] and [from the fall of 1962 through the spring of 1967] should '
                                         'be marked in their entirety, without separate markups for "November 2," '
                                         '"2001, "the fall," "1962," "the spring," and "1967." Dates that are part'
                                         ' of rate expressions such as "once per day" or "twice a year" '
                                         'should NOT be marked.',
                             vocabulary=None),
                      Entity(name='TIME',
                             description='Any time ending with "a.m." or "p.m." If the "a.m." or "p.m." is explicit, '
                                         'it must be tagged along with the numbers. Other times of day (units '
                                         'smaller than a day) and time durations less than 24 hours are also '
                                         'marked: morning, noon, night, 3 hours, a minute. Do not separate mentions'
                                         ' into their component parts: [the evening of July 4 th ] and [5:00am, '
                                         'April 5, 2008] should be marked in their entirety, without separate markups'
                                         ' for "evening," "July 4 th ," "5:00am," and "April 5, 2008"',
                             vocabulary=None),
                      Entity(name='PERCENT',
                             description='Any percentage. A percent symbol or the actual word percent must be '
                                         'explicit and included in the extent. If the percent is implicit, the '
                                         'number should be marked CARDINAL.',
                             vocabulary=None),
                      Entity(name='MONEY',
                             description='Any monetary value including all monetary denominations. The monetary unit'
                                         ' must be explicit and included in the extent. If the monetary unit is '
                                         'implicit, the number should be marked CARDINAL. Only values should be '
                                         'tagged—generic references to money should not. For example, in "money '
                                         'invested overseas," there is no markup for "money." In rate expressions'
                                         ' such as "$ per unit," the unit should not be included in the extent. '
                                         'For example, in "$3 per share," the extent is [$3].',
                             vocabulary=None),
                      Entity(name='QUANTITY',
                             description='Used to classify measurements with standardized units. If the unit of'
                                         ' measurement is implicit or non-standard (3 handfuls, 2 football fields,'
                                         ' 10 points), the number should be marked CARDINAL. One exception to this'
                                         ' rule is formulaic references to the age, height, and weight of a person:'
                                         ' Joe Smith, 44, five ten, two twenty. In this instance, [five ten] and'
                                         ' [two twenty] should be marked QUANTITY. (While [44] should be marked DATE)',
                             vocabulary=None),
                      Entity(name='ORDINAL', description='All ordinal numbers, including adverbials.', vocabulary=None),
                      Entity(name='CARDINAL',
                             description='Numerals, including whole numbers, fractions, and decimals, that provide'
                                         ' a count or quantity and do not fall under a unit of measurement, money,'
                                         ' percent, date or time. For "Nasdaq composite fill [1.39] to [451.37]." '
                                         'the numbers are marked CARDINAL because there is no monetary unit. '
                                         'Headless numerical phrases are also covered in this category: "reducing '
                                         'employment from [18,000] to [16,000]." Numbers identifying list items '
                                         'should also be included. Pronominal mentions of "one" should not be tagged.',
                             vocabulary=None),
                      Entity(name='EVENT',
                             description='Named hurricanes, battles, wars, sports events, attacks. Metonymic mentions '
                                         '(marked with a ~) of the date or location of an event, '
                                         'or of the organization(s) involved, are included.',
                             vocabulary=None),
                      Entity(name='WORK_OF_ART',
                             description='Titles of books, songs, television programs and other creations. '
                                         'Also includes awards. These are usually surrounded by quotation '
                                         'marks in the article (though the quotations are not included in '
                                         'the annotation). Newspaper headlines should only be marked if they '
                                         'are referential. In other words the headline of the article being '
                                         'annotated should not be marked but if in the body of the text here is '
                                         'a reference to an article, then it is markable as a work of art.',
                             vocabulary=None),
                      Entity(name='LAW',
                             description='Any document that has been made into a law, including named '
                                         'treaties and sections and chapters of named legal documents.',
                             vocabulary=None),
                      Entity(name='LANGUAGE', description='Any named language.', vocabulary=None)]
