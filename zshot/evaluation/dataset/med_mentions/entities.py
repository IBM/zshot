from zshot.utils.data_models import Entity

MEDMENTIONS_TYPE_INV = {
    'T058': "Health_Care_Activity", "T062": "Research_Activity", "T037": "Injury_or_Poisoning",
    "T038": "Biologic_Function", "T005": "Virus", "T007": "Bacterium", "T204": "Eukaryote",
    "T017": "Anotomical_Structure", "T074": "Medical_Device", "T031": "Body_Substance", "T103": "Chemical",
    "T168": "Food", "T201": "Clinical_Attribute", "T033": "Finding", "T082": "Spatial_Concept",
    "T022": "Body_System", "T091": "Biomedical_Occupation_or_Discipline", "T092": "Organization",
    "T097": "Professional_or_Occupational_Group", "T098": "Population_Group", "T170": "Intellectual_Product",
    "NEG": "NEG"
}

MEDMENTIONS_SPLITS = {
    "train": ['Biologic_Function', 'Chemical', 'Health_Care_Activity', 'Anotomical_Structure', "Finding",
              "Spatial_Concept", "Intellectual_Product", "Research_Activity", 'Medical_Device', 'Eukaryote',
              'Population_Group', 'NEG'],
    "validation": ['Biomedical_Occupation_or_Discipline', 'Virus', 'Clinical_Attribute', 'Injury_or_Poisoning',
                   'Organization', 'NEG'],
    "test": ['Body_System', 'Food', 'Body_Substance', 'Bacterium', 'Professional_or_Occupational_Group', 'NEG']
}

MEDMENTIONS_ENTITIES = [
    Entity(name="NEG",
           description="In this study, we fabticated prevascularized synthetic device ports to help mitigate this limitation. Thus, the optimum range of pore size for prevascularizaiton of these membranes was estimated to be 75 - 100 μm. A total of 51 patients were included, 16 in group I and 35 in group II."),
    Entity(name="T058",
           description="An activity of or relating to the practice of medicine or involving the care of patients."),
    Entity(name="T062", description="An activity carried out as part of research or experimentation."),
    Entity(name="T037", description="A traumatic wound, injury, or poisoning caused by an external agent or force."),
    Entity(name="T038", description="A state, activity or process of the body or one of its systems or parts."),
    Entity(name="T005",
           description="An organism consisting of a core of a single nucleic acid enclosed in a protective coat of protein. A virus may replicate only inside a host living cell. A virus exhibits some but not all of the usual characteristics of living things."),
    Entity(name="T007",
           description="A small, typically one-celled, prokaryotic micro-organism. Virtually all animal life on earth is dependent on bacteria for their survival as only bacteria and some archea possess the genes and enzymes necessary to synthesize vitamin B12, also known as cobalamin, and provide it through the food chain. Vitamin B12 is a water-soluble vitamin that is involved in the metabolism of every cell of the human body. It is a cofactor in DNA synthesis, and in both fatty acid and amino acid metabolism. It is particularly important in the normal functioning of the nervous system via its role in the synthesis of myelin."),
    Entity(name="T204",
           description="One of the three domains of life (the others being Bacteria and Archaea), also called Eukarya. These are organisms whose cells are enclosed in membranes and possess a nucleus. They comprise almost all multicellular and many unicellular organisms, and are traditionally divided into groups (sometimes called kingdoms) including Animals, Plants, Fungi, various Algae, and other taxa that were previously part of the old kingdom Protista."),
    Entity(name="T017",
           description="A normal or pathological part of the anatomy or structural organization of an organism."),
    Entity(name="T074",
           description="A manufactured object used primarily in the diagnosis, treatment, or prevention of physiologic or anatomic disorders."),
    Entity(name="T031",
           description="Extracellular material, or mixtures of cells and extracellular material, produced, excreted, or accreted by the body. Included here are substances such as saliva, dental enamel, sweat, and gastric acid."),
    Entity(name="T103",
           description="Compounds or substances of definite molecular composition. Chemicals are viewed from two distinct perspectives in the network, functionally and structurally. Almost every chemical concept is assigned at least two types, generally one from the structure hierarchy and at least one from the function hierarchy."),
    Entity(name="T168",
           description="Any substance generally containing nutrients, such as carbohydrates, proteins, and fats, that can be ingested by a living organism and metabolized into energy and body tissue. Some foods are naturally occurring, others are either partially or entirely made by humans."),
    Entity(name="T201",
           description="An observable or measurable property or state of an organism of clinical interest."),
    Entity(name="T033",
           description="That which is discovered by direct observation or measurement of an organism attribute or condition, including the clinical history of the patient. The history of the presence of a disease is a 'Finding' and is distinguished from the disease itself."),
    Entity(name="T082", description="A location, region, or space, generally having definite boundaries."),
    Entity(name="T022", description="A complex of anatomical structures that performs a common function."),
    Entity(name="T091", description="A vocation, academic discipline, or field of study related to biomedicine."),
    Entity(name="T092",
           description="The result of uniting for a common purpose or function. The continued existence of an organization is not dependent on any of its members, its location, or particular facility. Components or subparts of organizations are also included here. Although the names of organizations are sometimes used to refer to the buildings in which they reside, they are not inherently physical in nature."),
    Entity(name="T097", description="An individual or individuals classified according to their vocation."),
    Entity(name="T098",
           description="An indivdual or individuals classified according to their sex, racial origin, religion, common place of living, financial or social status, or some other cultural or behavioral attribute."),
    Entity(name="T170",
           description="A conceptual entity resulting from human endeavor. Concepts assigned to this type generally refer to information created by humans for some purpose.")
]
