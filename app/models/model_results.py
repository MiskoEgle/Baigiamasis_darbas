from app.models import db
from datetime import datetime

class ModelResults(db.Model):
    __tablename__ = 'model_results'
    
    id = db.Column(db.Integer, primary_key=True)
    modelio_tipas = db.Column(db.String(50), nullable=False)
    tikslumas = db.Column(db.Float, nullable=False)
    preciziškumas = db.Column(db.Float, nullable=False)
    atgaminimas = db.Column(db.Float, nullable=False)
    f1_balas = db.Column(db.Float, nullable=False)
    geriausias_k = db.Column(db.Integer, nullable=True)
    geriausias_metrika = db.Column(db.String(50), nullable=True)
    vidutinis_roc_auc = db.Column(db.Float, nullable=True)
    vidutinis_ap = db.Column(db.Float, nullable=True)
    sukūrimo_data = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ModelResults {self.modelio_tipas} {self.sukūrimo_data}>' 