# QuCardio Project Configuration
from pathlib import Path

# Class mappings
CLASS_NAMES = ['Normal', 'Arrhythmia', 'Myocardial_Infarction', 'History_of_MI']
CLASS_MAP = {'Normal': 0, 'Arrhythmia': 1, 'Myocardial_Infarction': 2, 'History_of_MI': 3}

# Image dimensions
IMAGE_SIZE = (340, 340)
RESNET_INPUT_SIZE = (224, 224)

# Quantum & Classical Model Hyperparameters
N_SVD_COMPONENTS = 9
FEATURE_DIMENSION = 9 # Number of SVD features to encode
N_QUBITS = 9          # Number of qubits (matches feature dimension)
REPS = 2              # Circuit repetitions for ZZFeatureMap
C_PARAM = 5.0         # QSVC regularization (tuned: best from sweep, paper=1.0)
PEGASOS_C = 1.0       # Pegasos C (paper value — Pegasos SGD requires small C)
TAU = 100             # For Pegasos QSVC
N_STEPS = 1000        # For Pegasos QSVC (paper uses 1000)

# Data splitting
TEST_SIZE = 0.2
RANDOM_SEED = 42

# Directories (adapted to the existing project structure)
DATA_RAW_DIR = Path('data/raw')
DATA_PROCESSED_DIR = Path('data/processed_340')
FEATURES_DIR = Path('data')
MODELS_DIR = Path('backend/models')
RESULTS_DIR = Path('results')
