# This file makes the app directory a Python package 

import os
import sys
import codecs
import json
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from dotenv import load_dotenv
from app.utils.logger import setup_logger
from app.models.database import db, init_db
from app.models.knn_model import KNNModelis
from app.models.cnn_model import CNNModelis
from app.models.transformer_model import TransformerModelis

# Užkrauti aplinkos kintamuosius
load_dotenv()

# Nustatome logging su UTF-8 koduote
logging.basicConfig(
    filename='app/static/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# Sukurti Flask aplikaciją
app = Flask(__name__)

# Konfigūruoti aplikaciją
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///kelio_zenklai.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'data/raw'
app.config['STATIC_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
app.config['TEMPLATES_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app.config['KNN_RESULTS_DIR'] = os.path.join(app.config['STATIC_FOLDER'], 'knn_grafikai')
app.config['CNN_RESULTS_DIR'] = os.path.join(app.config['STATIC_FOLDER'], 'cnn_grafikai')
app.config['TRANSFORMER_RESULTS_DIR'] = os.path.join(app.config['STATIC_FOLDER'], 'transformer_grafikai')

# Sukurti katalogus
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMPLATES_FOLDER'], exist_ok=True)
os.makedirs(app.config['KNN_RESULTS_DIR'], exist_ok=True)
os.makedirs(app.config['CNN_RESULTS_DIR'], exist_ok=True)
os.makedirs(app.config['TRANSFORMER_RESULTS_DIR'], exist_ok=True)

# Sukurti rezultatų katalogą
RESULTS_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
os.makedirs(RESULTS_FOLDER, exist_ok=True)
logging.info(f"Rezultatų katalogas sukurtas: {RESULTS_FOLDER}")

# Sukurti modelių katalogą
MODELS_FOLDER = 'app/static/models'
os.makedirs(MODELS_FOLDER, exist_ok=True)
logging.info("Modelių katalogas sukurtas")

# Patikriname ar egzistuoja aktyvus_modeliai.json
AKTYVUS_MODELIU_FAILAS = 'app/static/aktyvus_modeliai.json'
if not os.path.exists(AKTYVUS_MODELIU_FAILAS):
    pradinis = {
        "knn": "KNN_1",
        "cnn": "CNN_1",
        "transformer": "Transformer_1"
    }
    with codecs.open(AKTYVUS_MODELIU_FAILAS, 'w', encoding='utf-8') as f:
        json.dump(pradinis, f, ensure_ascii=False)
    logging.info("Sukurtas aktyvus_modeliai.json failas")

# Sukurti logger'į
logger = setup_logger()

# Inicializuoti duomenų bazę
db.init_app(app)
migrate = Migrate(app, db)
with app.app_context():
    db.create_all()
    logging.info("Duomenų bazė sėkmingai inicializuota")

# Inicializuoti modelius
try:
    knn_modelis = KNNModelis()
    cnn_modelis = CNNModelis()
    transformer_modelis = TransformerModelis()
    logging.info("Modeliai sėkmingai inicializuoti")
except Exception as e:
    logging.error(f"Klaida inicializuojant modelius: {str(e)}")
    raise

# Importuoti maršrutus
from app import routes

# Registruoti Blueprint
from .routes import bp
app.register_blueprint(bp)

# Importuoti modelius
from .models import *

# Importuoti įrankius
from .utils import *

# Globalus kintamasis treniravimo procesui
training_process = None

# sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
# sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# Čia galite pridėti kitą konfigūraciją, jei reikia 