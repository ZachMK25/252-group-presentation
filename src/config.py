"""
Configuration file for cryptocurrency prediction project
Contains all constants and hyperparameters used across scripts
"""

import numpy as np

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
DATA_FILE = 'crypto.csv'
FIELD_NAMES = {'open', 'high', 'low', 'close', 'volume'}

# Optional data filtering
KEEP_SYMBOLS = []  # Empty list = use all symbols
MAX_ROWS_PER_SYMBOL = None  # None = use all available rows

# ============================================================================
# FEATURE ENGINEERING PARAMETERS
# ============================================================================
# Rate of Change windows
ROC_WINS = [1, 3, 42]

# EMA pairs (fast, slow)
EMA_PAIRS = [(84, 168)]

# RSI windows
RSI_WINS = [8, 14, 26]

# CCI windows
CCI_WINS = [10, 20]

# Bollinger Bands windows
BB_WINS = [10, 20]

# Stochastic oscillator K values
STOCH_K_VALS = [8, 14]

# Volatility window
VOL_WIN = 20

# Volume windows
VOL_MEAN_WIN = 42
VOL_LOG_WINS = [42, 84]

# ============================================================================
# LABEL CREATION PARAMETERS
# ============================================================================
LABEL_THRESHOLD = 0.01  # 1% threshold for up/down classification

# ============================================================================
# CROSS-VALIDATION PARAMETERS
# ============================================================================
N_FOLDS = 4
EMBARGO_HOURS = 24  # Hours to purge between train/test splits

# ============================================================================
# MODEL PARAMETERS
# ============================================================================
USE_GPU_XGB = True  # Attempt GPU for XGBoost (fallback to CPU)
USE_HEAVY_MODELS = False  # Enable SVM and KNN (slower training)

# XGBoost parameters
XGB_N_ESTIMATORS = 300
XGB_LEARNING_RATE = 0.1
XGB_MAX_DEPTH = 6

# Random Forest parameters
RF_N_ESTIMATORS = 500
RF_MAX_DEPTH = None

# Decision Tree parameters
DT_MAX_DEPTH = 12

# Logistic Regression parameters
LR_C = 1.0
LR_PENALTY = 'l2'
LR_SOLVER = 'saga'
LR_MAX_ITER = 1000

# ============================================================================
# TRADE SIMULATION PARAMETERS
# ============================================================================
INITIAL_CAPITAL = 10000
TRANSACTION_COST = 0.001  # 0.1% per trade
THRESHOLD_MIN = 0.30
THRESHOLD_MAX = 0.80
THRESHOLD_STEP = 0.01

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================
MODEL_DIR = 'saved_models'
RESULTS_DIR = 'roi_results'
SIMULATION_DIR = 'simulation_results'
