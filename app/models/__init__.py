# This file makes the models directory a Python package 

import os
import sys
import codecs

# Importuoti modulius
from .knn_model import KNNModelis
from .cnn_model import CNNModelis
from .transformer_model import TransformerModelis
from .database import (
    KNNModelDB,
    CNNModelDB,
    TransformerModelDB,
    ModelResults,
    TestResult,
    init_db
)

# Sukurti katalogus jei neegzistuoja
os.makedirs('app/models/saved', exist_ok=True)

__all__ = [
    'KNNModelis',
    'CNNModelis',
    'TransformerModelis',
    'KNNModelDB',
    'CNNModelDB',
    'TransformerModelDB',
    'ModelResults',
    'TestResult',
    'init_db'
] 