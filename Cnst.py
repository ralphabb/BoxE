
DEFAULT_RANDOM_SEED = 20190503

SIGMOID = "SIGMOID"  
TANH = "TANH"
ELU = "ELU+log"
IDENTITY = "Identity"

POINTS = "Points"


UNIFORM = "UNIFORM"
BERNOULLI = "BERN"
GAN = "GAN"
SELFADV = "SELF-ADV"

CONSTRAINED_UNIF = "ConstrainedUnif"
UNIF = "Unif"

TRAIN = "Train"  
TEST = "Test"  

WRONG_MODE_WARNING = "Run Forward Passes in Test Mode to avoid Negative Sampling and Loss Computation"

LOSS_RANK = "lossrank"
DIST_RANK = "distrank"

SOFT = "softmin"
HARD = "hardmin"
SIGNSIG = "signsigmoid"

CATEGORICAL = "CAT"
RANKING = "RANK"
RAW = "RAW"
FILTERED = "FILTERED"
FACT_DELIMITER = "-"
DEFAULT_KB_DIR = "Datasets/"
DEFAULT_KB_MULTI_DIR = "DatasetsMulti/"
ENT_2_ID_DICT_NAME = "Ent2ID.dict"
REL_2_ID_DICT_NAME = "Rel2ID.dict"
HTR_KBs = ["FB15k", "WN18"]
KB_FORMAT = ".kb"
KBB_FORMAT = ".kbb"
KBB_WITH_NEG_FORMAT = ".nkbb"
KB_META_FILE_NAME = "Metadata.mpk"
KB_META_MULTI_FILE_NAME = "MetadataMulti.mpk"

NO_STOPS = "NN"
NO_STOP = "N"
STOP = "S"
POLY_LOSS = "poly"
Q2B_LOSS = "q2b"
STOP_SIZES = "NS"
STOP_POSITIONS = "SN"
STOP_BOTH = "SS"

VALID = "Valid"

NEG_SAMP = "negative sampling loss"
MARGIN_BASED = "margin-based loss"

AND = "&"
OR = "|"
IMPLIES = ">"
EQUIV = "="
TERMINAL = "Atom"

VIZ_TOOL_NAME = "BoxEViz"
