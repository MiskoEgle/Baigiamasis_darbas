import os
import sys
import codecs
import json
import logging
from dotenv import load_dotenv

# Užkrauti aplinkos kintamuosius
load_dotenv()

# Pagrindiniai katalogai
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Modelių katalogai
KNN_MODEL_DIR = os.path.join(STATIC_DIR, 'knn_grafikai')
CNN_MODEL_DIR = os.path.join(STATIC_DIR, 'cnn_grafikai')
TRANSFORMER_MODEL_DIR = os.path.join(STATIC_DIR, 'transformer_grafikai')

# Duomenų katalogai
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Rezultatų katalogai
KNN_RESULTS_DIR = os.path.join(RESULTS_DIR, 'knn')
CNN_RESULTS_DIR = os.path.join(RESULTS_DIR, 'cnn')
TRANSFORMER_RESULTS_DIR = os.path.join(RESULTS_DIR, 'transformer')

# Sukurti katalogus
for directory in [
    STATIC_DIR,
    DATA_DIR,
    RESULTS_DIR,
    KNN_MODEL_DIR,
    CNN_MODEL_DIR,
    TRANSFORMER_MODEL_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    KNN_RESULTS_DIR,
    CNN_RESULTS_DIR,
    TRANSFORMER_RESULTS_DIR
]:
    os.makedirs(directory, exist_ok=True)

# Modelių parametrai
KNN_PARAMS = {
    'n_neighbors': 5,
    'weights': 'uniform',
    'algorithm': 'auto',
    'leaf_size': 30,
    'p': 2,
    'metric': 'minkowski',
    'n_jobs': -1
}

CNN_PARAMS = {
    'input_shape': (32, 32, 3),
    'num_classes': 43,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50
}

TRANSFORMER_PARAMS = {
    'input_dim': 3072,  # 32x32x3
    'num_classes': 43,
    'd_model': 256,
    'nhead': 8,
    'num_layers': 6,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50
}

# Duomenų parametrai
DATA_PARAMS = {
    'test_size': 0.2,
    'val_size': 0.1,
    'augment': True,
    'target_size': (32, 32)
}

# Išsaugoti konfigūraciją
def save_config(path):
    """
    Išsaugoti konfigūraciją
    """
    try:
        config = {
            'KNN_PARAMS': KNN_PARAMS,
            'CNN_PARAMS': CNN_PARAMS,
            'TRANSFORMER_PARAMS': TRANSFORMER_PARAMS,
            'DATA_PARAMS': DATA_PARAMS
        }
        
        with codecs.open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
            
    except Exception as e:
        logging.error(f"Klaida išsaugant konfigūraciją: {str(e)}")
        raise
        
# Užkrauti konfigūraciją
def load_config(path):
    """
    Užkrauti konfigūraciją
    """
    try:
        with codecs.open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        return config
        
    except Exception as e:
        logging.error(f"Klaida užkraunant konfigūraciją: {str(e)}")
        raise

class Config:
    """
    Konfigūracijos klasė
    """
    def __init__(self):
        self.config_path = 'app/config.json'
        self.default_config = {
            'knn': {
                'n_neighbors': 5,
                'weights': 'uniform',
                'algorithm': 'auto',
                'leaf_size': 30,
                'p': 2,
                'metric': 'minkowski',
                'n_jobs': -1
            },
            'cnn': {
                'epochs': 10,
                'batch_size': 32,
                'learning_rate': 0.001,
                'validation_split': 0.2
            },
            'transformer': {
                'epochs': 10,
                'batch_size': 32,
                'learning_rate': 0.0001,
                'validation_split': 0.2
            },
            'data': {
                'image_size': (32, 32),
                'test_size': 0.2,
                'random_state': 42
            }
        }
        self.config = self.load_config()
        
    def load_config(self):
        """
        Užkrauti konfigūraciją iš failo
        """
        try:
            if os.path.exists(self.config_path):
                with codecs.open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logging.info("Konfigūracija sėkmingai užkrauta")
                return config
            else:
                logging.warning("Konfigūracijos failas nerastas, naudojama numatytoji konfigūracija")
                self.save_config(self.default_config)
                return self.default_config
                
        except Exception as e:
            logging.error(f"Klaida užkraunant konfigūraciją: {str(e)}")
            return self.default_config
            
    def save_config(self, config):
        """
        Išsaugoti konfigūraciją į failą
        """
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with codecs.open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            logging.info("Konfigūracija sėkmingai išsaugota")
            
        except Exception as e:
            logging.error(f"Klaida išsaugant konfigūraciją: {str(e)}")
            raise
            
    def get_knn_config(self):
        """
        Gauti KNN modelio konfigūraciją
        """
        return self.config.get('knn', self.default_config['knn'])
        
    def get_cnn_config(self):
        """
        Gauti CNN modelio konfigūraciją
        """
        return self.config.get('cnn', self.default_config['cnn'])
        
    def get_transformer_config(self):
        """
        Gauti Transformer modelio konfigūraciją
        """
        return self.config.get('transformer', self.default_config['transformer'])
        
    def get_data_config(self):
        """
        Gauti duomenų konfigūraciją
        """
        return self.config.get('data', self.default_config['data'])
        
    def update_config(self, section, key, value):
        """
        Atnaujinti konfigūraciją
        """
        try:
            if section in self.config:
                self.config[section][key] = value
                self.save_config(self.config)
                logging.info(f"Konfigūracija atnaujinta: {section}.{key} = {value}")
            else:
                logging.error(f"Konfigūracijos sekcija nerasta: {section}")
                
        except Exception as e:
            logging.error(f"Klaida atnaujinant konfigūraciją: {str(e)}")
            raise 