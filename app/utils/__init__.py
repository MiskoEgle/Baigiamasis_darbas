import os
import sys
# import codecs

# Nustatyti UTF-8 koduotę
# sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
# sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# Importuoti modulius
from .config import *
from .logger import *
from .data_processing import *
from .dataset_loader import *
from .visualization import *

# Sukurti katalogus jei neegzistuoja
os.makedirs('app/static/test_images', exist_ok=True)
os.makedirs('app/static/knn_grafikai', exist_ok=True)
os.makedirs('app/static/cnn_grafikai', exist_ok=True)
os.makedirs('app/static/transformer_grafikai', exist_ok=True)
os.makedirs('data/raw', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Sukurti logger'į
logger = setup_logger()

# Sukurti konfigūraciją
config = Config()

__all__ = [
    'Config',
    'setup_logger',
    'log_info',
    'log_error',
    'log_debug',
    'log_warning',
    'log_critical',
    'process_image',
    'augment_image',
    'prepare_data',
    'save_data',
    'load_data',
    'load_dataset',
    'save_dataset_info',
    'load_dataset_info',
    'sukurti_konfuzijos_macica',
    'sukurti_klasiu_grafikus',
    'sukurti_metriku_grafikus',
    'issaugoti_rezultatus',
    'logger',
    'config'
] 