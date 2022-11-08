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
              'Population_Group'],
    "validation": ['Biomedical_Occupation_or_Discipline', 'Virus', 'Clinical_Attribute', 'Injury_or_Poisoning',
                   'Organization'],
    "test": ['Body_System', 'Food', 'Body_Substance', 'Bacterium', 'Professional_or_Occupational_Group']}

MEDMENTIONS_ENTITIES = [
    Entity(name="T058", description="Healthcare activity"),
    Entity(name="T062", description="Research activity"),
    Entity(name="T037", description="Injury or Poisoning"),
    Entity(name="T038", description="Biologic Function"),
    Entity(name="T005", description="Virus"),
    Entity(name="T007", description="Bacterium"),
    Entity(name="T204", description="Eukaryote"),
    Entity(name="T017", description="Anotomical structure"),
    Entity(name="T074", description="Medical device"),
    Entity(name="T031", description="Body substance"),
    Entity(name="T103", description="Chemical"),
    Entity(name="T168", description="Food"),
    Entity(name="T201", description="Clinical Attribute"),
    Entity(name="T033", description="Finding"),
    Entity(name="T082", description="Spatial Concept"),
    Entity(name="T022", description="Body System"),
    Entity(name="T091", description="Biomedical occupation or discipline"),
    Entity(name="T092", description="Organization"),
    Entity(name="T097", description="Professional or occupational group"),
    Entity(name="T098", description="Population group"),
    Entity(name="T170", description="Intellectual product")
]
