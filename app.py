from flask import Flask, render_template, request, jsonify
import os
from app.models.database import init_db, db
from app.models.knn_model import KNNModelis
from app.models.cnn_model import CNNModelis
from app.models.transformer_model import VizijosTransformeris
from app.utils.data_processing import process_image

app = Flask(__name__, template_folder='app/templates')
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///kelio_zenklai.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'data/raw'

# Inicializuoti duomenų bazę
init_db(app)

# Inicializuoti modelius
knn_modelis = KNNModelis()
cnn_modelis = CNNModelis()
transformer_modelis = VizijosTransformeris()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mokyti', methods=['POST'])
def train():
    if 'file' not in request.files:
        return jsonify({'klaida': 'Nepateiktas failas'}), 400
    
    failas = request.files['file']
    modelio_tipas = request.form.get('modelio_tipas', 'knn')
    
    if failas.filename == '':
        return jsonify({'klaida': 'Nepasirinktas failas'}), 400
    
    # Išsaugoti įkeltą failą
    failo_pavadinimas = os.path.join(app.config['UPLOAD_FOLDER'], failas.filename)
    failas.save(failo_pavadinimas)
    
    # Apdoroti paveikslėlį ir mokyti modelį
    apdorotas_paveikslėlis = process_image(failo_pavadinimas)
    
    if modelio_tipas == 'knn':
        rezultatai = knn_modelis.mokyti(apdorotas_paveikslėlis)
    elif modelio_tipas == 'cnn':
        rezultatai = cnn_modelis.mokyti(apdorotas_paveikslėlis)
    elif modelio_tipas == 'transformer':
        rezultatai = transformer_modelis.mokyti(apdorotas_paveikslėlis)
    else:
        return jsonify({'klaida': 'Neteisingas modelio tipas'}), 400
    
    return jsonify(rezultatai)

@app.route('/prognozuoti', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'klaida': 'Nepateiktas failas'}), 400
    
    failas = request.files['file']
    modelio_tipas = request.form.get('modelio_tipas', 'knn')
    
    if failas.filename == '':
        return jsonify({'klaida': 'Nepasirinktas failas'}), 400
    
    # Išsaugoti įkeltą failą
    failo_pavadinimas = os.path.join(app.config['UPLOAD_FOLDER'], failas.filename)
    failas.save(failo_pavadinimas)
    
    # Apdoroti paveikslėlį ir padaryti prognozę
    apdorotas_paveikslėlis = process_image(failo_pavadinimas)
    
    if modelio_tipas == 'knn':
        prognozė = knn_modelis.prognozuoti(apdorotas_paveikslėlis)
    elif modelio_tipas == 'cnn':
        prognozė = cnn_modelis.prognozuoti(apdorotas_paveikslėlis)
    elif modelio_tipas == 'transformer':
        prognozė = transformer_modelis.prognozuoti(apdorotas_paveikslėlis)
    else:
        return jsonify({'klaida': 'Neteisingas modelio tipas'}), 400
    
    return jsonify({'prognozė': prognozė})

@app.route('/rezultatai')
def results():
    return render_template('results.html')

@app.route('/informacija')
def informacija():
    return render_template('informacija.html')

@app.route('/naudojimas')
def naudojimas():
    return render_template('naudojimas.html')

if __name__ == '__main__':
    app.run(debug=True) 