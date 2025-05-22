from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import logging
import codecs

# Create a single SQLAlchemy instance
db = SQLAlchemy()

def init_db(app):
    """Initialize the database with the Flask app"""
    try:
        # Initialize SQLAlchemy with the app
        db.init_app(app)
        
        with app.app_context():
            db.create_all()
            logging.info("Duomenų bazė sėkmingai inicializuota")
    except Exception as e:
        logging.error(f"Klaida inicializuojant duomenų bazę: {str(e)}")
        raise

class TrafficSign(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    class_id = db.Column(db.Integer, nullable=False)
    width = db.Column(db.Integer)
    height = db.Column(db.Integer)
    roi_x1 = db.Column(db.Integer)
    roi_y1 = db.Column(db.Integer)
    roi_x2 = db.Column(db.Integer)
    roi_y2 = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ModelResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_type = db.Column(db.String(50), nullable=False)
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    confusion_matrix = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class UserTest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(50), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    predicted_class = db.Column(db.Integer)
    confidence = db.Column(db.Float)
    model_type = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ModelHyperparameters(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_type = db.Column(db.String(50), nullable=False)
    parameters = db.Column(db.Text, nullable=False)
    accuracy = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ModelResults(db.Model):
    __tablename__ = 'modelio_rezultatai'
    id = db.Column(db.Integer, primary_key=True)
    cnn_modelio_id = db.Column(db.Integer, db.ForeignKey('cnn_modeliai.id'))
    knn_modelio_id = db.Column(db.Integer, db.ForeignKey('knn_modeliai.id'))
    transformer_modelio_id = db.Column(db.Integer, db.ForeignKey('transformer_modeliai.id'))
    tikslumas = db.Column(db.Float)
    preciziskumas = db.Column(db.Float)
    atgaminimas = db.Column(db.Float)
    f1_balas = db.Column(db.Float)
    data = db.Column(db.DateTime, default=datetime.now)

class TestResult(db.Model):
    __tablename__ = 'test_rezultatai'
    
    id = db.Column(db.Integer, primary_key=True)
    modelio_tipas = db.Column(db.String(50), nullable=False)
    modelio_pavadinimas = db.Column(db.String(100))
    paveiksliuko_kelias = db.Column(db.String(200))
    prognozuota_klase = db.Column(db.Integer)
    atpazintas_zenklas = db.Column(db.Integer)
    zenklo_pavadinimas = db.Column(db.String(100))
    tikimybe = db.Column(db.Float)
    tikroji_klase = db.Column(db.Integer)
    tikslumas = db.Column(db.Float)
    data = db.Column(db.DateTime, default=datetime.now)

class CNNModelDB(db.Model):
    __tablename__ = 'cnn_modeliai'
    
    id = db.Column(db.Integer, primary_key=True)
    pavadinimas = db.Column(db.String(100), unique=True, nullable=False)
    sukurtas = db.Column(db.DateTime, default=datetime.now)
    modelio_duomenys = db.Column(db.LargeBinary)
    rezultatai = db.relationship('ModelResults', backref='cnn_modelis', lazy=True)

class TransformerModelDB(db.Model):
    __tablename__ = 'transformer_modeliai'
    
    id = db.Column(db.Integer, primary_key=True)
    pavadinimas = db.Column(db.String(100), unique=True, nullable=False)
    sukurtas = db.Column(db.DateTime, default=datetime.now)
    modelio_duomenys = db.Column(db.LargeBinary)
    rezultatai = db.relationship('ModelResults', backref='transformer_modelis', lazy=True, primaryjoin="TransformerModelDB.id==ModelResults.transformer_modelio_id")

class KNNModelDB(db.Model):
    __tablename__ = 'knn_modeliai'
    id = db.Column(db.Integer, primary_key=True)
    pavadinimas = db.Column(db.String(100), unique=True, nullable=False)
    k_reiksme = db.Column(db.Integer)
    sukurtas = db.Column(db.DateTime, default=datetime.now)
    modelio_duomenys = db.Column(db.LargeBinary)
    rezultatai = db.relationship('ModelResults', backref='knn_modelis', lazy=True, primaryjoin="KNNModelDB.id==ModelResults.knn_modelio_id") 