import os
import sys
import codecs
import json
import logging
import traceback
from flask import (
    Blueprint,
    render_template,
    request,
    jsonify,
    current_app,
    send_from_directory,
    redirect,
    url_for
)
from werkzeug.utils import secure_filename
from .models import (
    KNNModelis,
    CNNModelis,
    TransformerModelis,
    TestResult,
)
from app.models.database import db
from .utils import (
    process_image,
    augment_image,
    prepare_data,
    load_dataset,
    save_dataset_info,
    load_dataset_info,
    sukurti_konfuzijos_macica,
    sukurti_klasiu_grafikus,
    sukurti_metriku_grafikus,
    issaugoti_rezultatus
)
from app.utils.zenklu_zodynas import ZENKLU_ZODYNAS
from app.utils.dataset_loader import nuskaityti_is_csv, nuskaityti_is_zip
from app.utils.data_processing import universalus_duomenu_nuskaitymas, load_dataset_cnn

from app.model_progress import model_progress

# Sukurti Blueprint
bp = Blueprint('main', __name__)

# Sukurti modelius
knn_model = KNNModelis()
cnn_model = CNNModelis()
transformer_model = TransformerModelis()

# Pagrindinis puslapis
@bp.route('/')
def index():
    # Surinkti modelių sąrašus kaip ir rezultatų puslapyje
    knn_modeliai, cnn_modeliai, transformer_modeliai = [], [], []
    models_dir = 'app/static/models'
    if os.path.exists(models_dir):
        for fname in os.listdir(models_dir):
            if fname.endswith('.joblib') or fname.endswith('.joblib.joblib'):
                if fname.startswith('knn_'):
                    knn_modeliai.append({'pavadinimas': fname})
                elif fname.startswith('cnn_'):
                    cnn_modeliai.append({'pavadinimas': fname})
                elif fname.startswith('transformer_'):
                    transformer_modeliai.append({'pavadinimas': fname})
    return render_template('index.html',
                          knn_modeliai=knn_modeliai,
                          cnn_modeliai=cnn_modeliai,
                          transformer_modeliai=transformer_modeliai)

# Mokymo puslapis
@bp.route('/mokymas')
def mokyti():
    return render_template('mokymas.html')

# Testavimo puslapis
@bp.route('/testuoti', methods=['GET', 'POST'])
def testuoti():
    if request.method == 'POST':
        try:
            # Gauti duomenis iš formos
            image = request.files['image']
            model_type = request.form['model_type']
            
            # Išsaugoti paveikslėlį
            filename = secure_filename(image.filename)
            image_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            image.save(image_path)
            
            # Apdoroti paveikslėlį
            processed_image = process_image(image_path)
            
            # Prognozuoti
            if model_type == 'knn':
                model = knn_model
            elif model_type == 'cnn':
                model = cnn_model
            else:
                model = transformer_model
                
            prediction = model.prognozuoti(processed_image)
            
            return jsonify({
                'success': True,
                'prediction': int(prediction),
                'confidence': float(model.confidence)
            })
            
        except Exception as e:
            logging.error(f"Klaida testuojant modelį: {str(e)}")
            return jsonify({'success': False, 'message': str(e)})
            
    return render_template('testuoti.html')

# Rezultatų puslapis
@bp.route('/rezultatai')
def rezultatai():
    try:
        # Surinkti KNN modelius
        knn_modeliai = []
        models_dir = 'app/static/models'
        if os.path.exists(models_dir):
            for fname in os.listdir(models_dir):
                if fname.endswith('.joblib') or fname.endswith('.joblib.joblib'):
                    if fname.startswith('knn_'):
                        knn_modeliai.append({'pavadinimas': fname})
        
        # Surinkti KNN rezultatus
        knn_results = []
        knn_results_dir = os.path.join('app', 'results', 'knn')
        if os.path.exists(knn_results_dir):
            for filename in os.listdir(knn_results_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(knn_results_dir, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                    result['failo_pavadinimas'] = filename
                    knn_results.append(result)
        
        # Gauti testavimo rezultatus
        query = TestResult.query.filter_by(modelio_tipas='knn')
        test_results = query.order_by(TestResult.data.desc()).all()
        
        return render_template('rezultatai_knn.html', 
                             knn_results=knn_results, 
                             knn_modeliai=knn_modeliai,
                             test_results=test_results)
    except Exception as e:
        return render_template('rezultatai_knn.html', 
                             knn_results=[], 
                             knn_modeliai=[], 
                             test_results=[],
                             klaida=str(e))

# Progreso puslapis
@bp.route('/progresas')
def progresas():
    return jsonify(model_progress.get_all())

# Statinių failų puslapis
@bp.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(current_app.config['STATIC_FOLDER'], filename)

@bp.route('/grafikai')
def grafikai():
    modelio_tipas = request.args.get('modelis', 'knn')
    katalogai = {
        'knn': 'knn_grafikai',
        'cnn': 'cnn_grafikai',
        'transformer': 'transformer_grafikai'
    }
    katalogas = katalogai.get(modelio_tipas, 'knn_grafikai')
    pilnas_katalogas = os.path.join('app/static', katalogas)
    if not os.path.exists(pilnas_katalogas):
        grafikai = []
    else:
        grafikai = [f for f in os.listdir(pilnas_katalogas) if f.endswith('.png')]
    grafikai = [f'{katalogas}/{f}' for f in grafikai]

    # Surinkti rezultatų failų sąrašus
    knn_dir = os.path.join('app', 'results', 'knn')
    cnn_dir = os.path.join('app', 'results', 'cnn')
    transformer_dir = os.path.join('app', 'results', 'transformer')
    knn_files = [f for f in os.listdir(knn_dir) if f.endswith('.json')] if os.path.exists(knn_dir) else []
    cnn_files = [f for f in os.listdir(cnn_dir) if f.endswith('.json')] if os.path.exists(cnn_dir) else []
    transformer_files = [f for f in os.listdir(transformer_dir) if f.endswith('.json')] if os.path.exists(transformer_dir) else []

    return render_template(
        'grafikai.html',
        grafikai=grafikai,
        modelio_tipas=modelio_tipas,
        knn_files=knn_files,
        cnn_files=cnn_files,
        transformer_files=transformer_files
    )

@bp.route('/json-failai')
def json_failai():
    failai = [f for f in os.listdir('app/static') if f.endswith('.json')]
    return render_template('json_failai.html', failai=failai)

@bp.route('/perziureti_json_faila/<failo_pavadinimas>')
def perziureti_json_faila(failo_pavadinimas):
    try:
        with open(os.path.join('app/static', failo_pavadinimas), 'r', encoding='utf-8') as f:
            turinys = json.load(f)
        return render_template('json_perziura.html', failo_pavadinimas=failo_pavadinimas, turinys=turinys)
    except Exception as e:
        return render_template('json_perziura.html', klaida=str(e))

@bp.route('/importuoti_rezultatus')
def importuoti_rezultatus():
    try:
        import json
        from app.models.knn_model import KNNModelis
        import os
        failas = 'app/static/knn_porciju_rezultatai.json'
        if not os.path.exists(failas):
            return render_template('importuoti_rezultatus.html', message='Rezultatų failas nerastas.'), 404
        with open(failas, encoding='utf-8') as f:
            rezultatai = json.load(f)
        # Palaikyti tiek sąrašą, tiek žodyną
        if isinstance(rezultatai, dict):
            rezultatai = [rezultatai]
        modelis = KNNModelis()
        kiek = 0
        with current_app.app_context():
            for res in rezultatai:
                try:
                    if not isinstance(res, dict):
                        continue  # praleisti, jei ne žodynas
                    metrikos = res.get('kitos_metrikos', {})
                    metrikos['tikslumas'] = res.get('tikslumas', 0.0)
                    metrikos['preciziškumas'] = res.get('preciziskumas', 0.0)
                    metrikos['atgaminimas'] = res.get('atgaminimas', 0.0)
                    metrikos['f1_balas'] = res.get('f1_balas', 0.0)
                    modelis.issaugoti_rezultatus(metrikos)
                    kiek += 1
                except Exception as e:
                    current_app.logger.error(f"Klaida įrašant rezultatą: {str(e)}")
                    continue
        return render_template('importuoti_rezultatus.html', message=f'Į DB importuota rezultatų: {kiek}')
    except Exception as e:
        current_app.logger.error(f"Klaida importuojant rezultatus: {str(e)}")
        return render_template('importuoti_rezultatus.html', message=f'Įvyko klaida: {str(e)}'), 500 

@bp.route('/informacija')
def informacija():
    return render_template('informacija.html')

@bp.route('/naudojimas')
def naudojimas():
    return render_template('naudojimas.html')

@bp.route('/prognoze', methods=['POST'])
def prognoze():
    try:
        if 'file' not in request.files:
            return jsonify({'klaida': 'Nepateiktas failas'}), 400
        failas = request.files['file']
        modelio_tipas = request.form.get('modelio_tipas', 'knn')
        # Get the correct modelio_pavadinimas field depending on modelio_tipas
        if modelio_tipas == 'knn':
            modelio_pavadinimas = request.form.get('modelio_pavadinimas_knn')
        elif modelio_tipas == 'cnn':
            modelio_pavadinimas = request.form.get('modelio_pavadinimas_cnn')
        elif modelio_tipas == 'transformer':
            modelio_pavadinimas = request.form.get('modelio_pavadinimas_transformer')
        else:
            modelio_pavadinimas = None
        if failas.filename == '':
            return jsonify({'klaida': 'Nepasirinktas failas'}), 400

        # Išsaugome paveiksliuką
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        paveiksliuko_kelias = os.path.join('app/static/test_images', f'test_{timestamp}_{failas.filename}')
        os.makedirs(os.path.dirname(paveiksliuko_kelias), exist_ok=True)
        failas.save(paveiksliuko_kelias)

        apdorotas_paveikslėlis = process_image(paveiksliuko_kelias)
        # KNN atveju: RGB, 32x32, flatten (3072 features)
        if modelio_tipas == 'knn':
            import numpy as np
            import cv2
            if apdorotas_paveikslėlis.shape[-1] == 1:
                # Grayscale -> RGB
                apdorotas_paveikslėlis = np.repeat(apdorotas_paveikslėlis, 3, axis=-1)
            apdorotas_paveikslėlis = cv2.resize((apdorotas_paveikslėlis * 255).astype(np.uint8), (32, 32))
            apdorotas_paveikslėlis = apdorotas_paveikslėlis.flatten().reshape(1, -1) / 255.0
        # CNN/transformer atveju paliekame kaip yra
        
        # Gauname modelio failo pavadinimą
        if modelio_tipas == 'knn' and modelio_pavadinimas:
            if modelio_pavadinimas.endswith('.joblib'):
                modelio_failas = f'app/static/models/{modelio_pavadinimas}'
            else:
                modelio_failas = f'app/static/models/knn_{modelio_pavadinimas}.joblib'
        elif modelio_tipas == 'knn':
            modelio_failas = 'app/static/models/knn_model.joblib'
        elif modelio_tipas == 'cnn' and modelio_pavadinimas:
            logger = logging.getLogger('app')
            logger.info(f'Gautas modelio pavadinimas iš formos: {modelio_pavadinimas}')
            modelio_failas = f'app/static/models/{modelio_pavadinimas}'
            logger.info(f'Naudojamas modelio failas: {modelio_failas}')
            katalogo_failai = os.listdir('app/static/models')
            logger.info(f'Failai kataloge app/static/models/: {katalogo_failai}')
            try:
                if cnn_model.model is None:
                    if os.path.exists(modelio_failas):
                        logger.info(f'Failas rastas: {modelio_failas}')
                        cnn_model.uzsikrauti_modeli(modelio_failas)
                        logger.info('Modelis įkeltas į atmintį.')
                    else:
                        logger.error(f'Failas nerastas: {modelio_failas}')
                        logger.error(f'Failai kataloge: {katalogo_failai}')
                        return jsonify({'klaida': f'Modelio failas nerastas: {modelio_failas}'}), 400
                else:
                    logger.info('Modelis jau įkeltas į atmintį.')
            except Exception as e:
                logger.error(f'Klaida įkeliant modelį: {str(e)}')
                logger.error(f'Failai kataloge: {katalogo_failai}')
                return jsonify({'klaida': f'Klaida įkeliant modelį: {str(e)}'}), 500
            prognozė, tikimybė = cnn_model.prognozuoti(apdorotas_paveikslėlis)
        elif modelio_tipas == 'transformer' and modelio_pavadinimas:
            if modelio_pavadinimas.endswith('.joblib'):
                modelio_failas = f'app/static/models/{modelio_pavadinimas}'
            else:
                modelio_failas = f'app/static/models/transformer_{modelio_pavadinimas}.joblib'
        elif modelio_tipas == 'transformer':
            modelio_failas = 'app/static/models/transformer_model.pt'
        else:
            return jsonify({'klaida': 'Neteisingas modelio tipas'}), 400

        # Užkrauname modelį
        if modelio_tipas == 'knn':
            from app.models.knn_model import KNNModelis
            knn_modelis = KNNModelis()
            if os.path.exists(modelio_failas):
                import joblib
                knn_modelis.model = joblib.load(modelio_failas)
            else:
                return jsonify({'klaida': 'Modelis nerastas'}), 400
            prognozė, tikimybė = knn_modelis.prognozuoti(apdorotas_paveikslėlis)
        elif modelio_tipas == 'cnn':
            prognozė, tikimybė = cnn_model.prognozuoti(apdorotas_paveikslėlis)
        elif modelio_tipas == 'transformer':
            modelio_failas = 'app/static/models/transformer_model.pt'
            if transformer_model.model is None:
                if os.path.exists(modelio_failas):
                    transformer_model.uzsikrauti_modeli(modelio_failas)
                else:
                    try:
                        transformer_model.uzsikrauti_is_db()
                    except Exception:
                        return jsonify({'klaida': 'Modelis dar nebuvo išmokytas ir nėra išsaugotas.'}), 400
            prognozė, tikimybė = transformer_model.prognozuoti(apdorotas_paveikslėlis)
        else:
            return jsonify({'klaida': 'Neteisingas modelio tipas'}), 400

        # Išsaugome rezultatą duomenų bazėje
        zenklo_pavadinimas = ZENKLU_ZODYNAS.get(int(prognozė), str(prognozė))
        test_result = TestResult(
            modelio_tipas=modelio_tipas,
            modelio_pavadinimas=modelio_pavadinimas,
            atpazintas_zenklas=int(prognozė),
            zenklo_pavadinimas=zenklo_pavadinimas,
            paveiksliuko_kelias=paveiksliuko_kelias,
            tikimybe=float(tikimybė)
        )
        db.session.add(test_result)
        db.session.commit()

        original_img_path = None
        if prognozė and prognozė is not None:
            # 1. Ieškome static/meta/
            static_meta_dir = os.path.join('app', 'static', 'meta')
            for ext in ['png', 'jpg', 'jpeg']:
                candidate = os.path.join(static_meta_dir, f"{prognozė}.{ext}")
                if os.path.exists(candidate):
                    original_img_path = f"meta/{prognozė}.{ext}"
                    break
            # 2. Jei nerado static/meta/, bandome duomenys/Meta/ (debugui)
            if not original_img_path:
                meta_dir = os.path.join('duomenys', 'Meta')
                for ext in ['png', 'jpg', 'jpeg']:
                    candidate = os.path.join(meta_dir, f"{prognozė}.{ext}")
                    if os.path.exists(candidate):
                        original_img_path = candidate
                        break

        # Vietoj JSON - redirect į vizualizacijos puslapį
        return redirect(url_for('main.testo_vizualizacija'))
    except FileNotFoundError as e:
        return jsonify({'klaida': f'Failas nerastas: {str(e)}'}), 400
    except ValueError as e:
        return jsonify({'klaida': f'Neteisingi duomenys: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'klaida': f'Įvyko nenumatyta klaida: {str(e)}'}), 500 

@bp.route('/testo_vizualizacija')
def testo_vizualizacija():
    # Paimti paskutinį test_result iš DB
    test_result = TestResult.query.order_by(TestResult.data.desc()).first()
    original_img_path = None
    if test_result and test_result.atpazintas_zenklas is not None:
        static_meta_dir = os.path.join('app', 'static', 'meta')
        for ext in ['png', 'jpg', 'jpeg']:
            candidate = os.path.join(static_meta_dir, f"{test_result.atpazintas_zenklas}.{ext}")
            if os.path.exists(candidate):
                original_img_path = f"meta/{test_result.atpazintas_zenklas}.{ext}"
                break
        if not original_img_path:
            meta_dir = os.path.join('duomenys', 'Meta')
            for ext in ['png', 'jpg', 'jpeg']:
                candidate = os.path.join(meta_dir, f"{test_result.atpazintas_zenklas}.{ext}")
                if os.path.exists(candidate):
                    original_img_path = candidate
                    break
    return render_template('testo_vizualizacija.html', test_result=test_result, original_img_path=original_img_path)

@bp.route('/train', methods=['POST'])
def train():
    try:
        print('request.form:', dict(request.form))
        print('request.files:', dict(request.files))
        model_type = request.form.get('modelio_tipas', request.form.get('model_type', 'cnn'))
        modelio_pavadinimas = request.form.get('modelio_pavadinimas', 'modelis')
        augment = request.form.get('augment', 'false').lower() == 'true'
        mokymo_tipas = request.form.get('mokymo_tipas', 'naujas')

        # Tikrinti ar įkeltas failas
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            filename = secure_filename(file.filename)
            save_path = os.path.join('duomenys', filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            file.save(save_path)
            csv_path = save_path
            data_dir = 'duomenys'  # katalogas, kuriame yra Train/...
        else:
            # Naudoti numatytąjį failą
            csv_path = 'data/raw/annotations.csv'
            data_dir = 'data/raw'

        print('Kviečiu load_dataset su:', data_dir, csv_path)
        if model_type == 'cnn':
            print('Naudoju load_dataset_cnn CNN modeliui')
            X, y = load_dataset_cnn(data_dir, csv_path)
        else:
            X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = load_dataset(data_dir, test_size=0.2)
            X, y = X_train, y_train
        print('Po load_dataset:', type(X), getattr(X, 'shape', None), type(y), getattr(y, 'shape', None))
        print('Po load_dataset: X example:', X[:1] if hasattr(X, '__getitem__') else X)
        print('Po load_dataset: y example:', y[:5] if hasattr(y, '__getitem__') else y)

        # Paruošti duomenis
        X_processed, y_processed = prepare_data(X, y, augment=augment, model_type=model_type)
        print('Po prepare_data: X_processed type:', type(X_processed), 'X_processed shape:', getattr(X_processed, 'shape', None))
        print('Po prepare_data: y_processed type:', type(y_processed), 'y_processed shape:', getattr(y_processed, 'shape', None))
        print('Po prepare_data: y_processed example:', y_processed[:5] if hasattr(y_processed, '__getitem__') else y_processed)

        # Mokyti modelį
        if model_type == 'knn':
            rezultatai = knn_model.mokyti(X_processed, y_processed, mokymo_tipas=mokymo_tipas)
        elif model_type == 'cnn':
            rezultatai = cnn_model.mokyti(X_processed, y_processed, mokymo_tipas=mokymo_tipas)
        else:
            rezultatai = transformer_model.mokyti(X_processed, y_processed)

        return jsonify({'success': True, 'results': rezultatai})

    except Exception as e:
        logging.error(f'Klaida mokant modelį: {str(e)}')
        return jsonify({'success': False, 'message': str(e)})

@bp.route('/stop_training', methods=['POST'])
def stop_training():
    try:
        model_type = request.form.get('model_type')
        if not model_type or model_type not in ['knn', 'cnn', 'transformer']:
            return jsonify({'klaida': 'Neteisingas modelio tipas'}), 400
            
        model_progress.stop(model_type)
        return jsonify({'message': f'Stopping {model_type} training...'})
    except Exception as e:
        return jsonify({'klaida': str(e)}), 500 

@bp.route('/prognoze_rezultatas')
def prognoze_rezultatas():
    from app.models.database import TestResult
    import os
    prognoze = TestResult.query.order_by(TestResult.data.desc()).first()
    original_img_path = None
    if prognoze and prognoze.atpazintas_zenklas is not None:
        # 1. Ieškome static/meta/
        static_meta_dir = os.path.join('app', 'static', 'meta')
        for ext in ['png', 'jpg', 'jpeg']:
            candidate = os.path.join(static_meta_dir, f"{prognoze.atpazintas_zenklas}.{ext}")
            if os.path.exists(candidate):
                original_img_path = f"meta/{prognoze.atpazintas_zenklas}.{ext}"
                break
        # 2. Jei nerado static/meta/, bandome duomenys/Meta/ (debugui)
        if not original_img_path:
            meta_dir = os.path.join('duomenys', 'Meta')
            for ext in ['png', 'jpg', 'jpeg']:
                candidate = os.path.join(meta_dir, f"{prognoze.atpazintas_zenklas}.{ext}")
                if os.path.exists(candidate):
                    original_img_path = candidate
                    break
    return render_template('prognoze.html', prognoze=prognoze, original_img_path=original_img_path)

@bp.route('/rezultatai_cnn')
def rezultatai_cnn():
    try:
        cnn_modeliai = []
        models_dir = 'app/static/models'
        if os.path.exists(models_dir):
            for fname in os.listdir(models_dir):
                if (fname.endswith('.joblib') or fname.endswith('.joblib.joblib') or fname.endswith('.h5')) and fname.startswith('cnn'):
                    cnn_modeliai.append({'pavadinimas': fname})
        cnn_results = []
        cnn_results_dir = os.path.join('app', 'results', 'cnn')
        eng2lt = {
            'accuracy': 'tikslumas',
            'precision': 'preciziškumas',
            'recall': 'atgaminimas',
            'f1': 'f1_balas',
        }
        if os.path.exists(cnn_results_dir):
            for filename in os.listdir(cnn_results_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(cnn_results_dir, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                    result['failo_pavadinimas'] = filename
                    # Metrikų raktų pervadinimas
                    if 'metrikos' in result and isinstance(result['metrikos'], dict):
                        for eng, lt in eng2lt.items():
                            if eng in result['metrikos']:
                                result['metrikos'][lt] = result['metrikos'][eng]
                    # Pridėti datą, jei jos nėra
                    if not result.get('data'):
                        import datetime
                        mtime = os.path.getmtime(file_path)
                        result['data'] = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                    # Pridėti duomenų failo pavadinimą, jei jo nėra
                    if not result.get('duomenu_failas'):
                        result['duomenu_failas'] = filename
                    cnn_results.append(result)
        query = TestResult.query.filter_by(modelio_tipas='cnn')
        test_results = query.order_by(TestResult.data.desc()).all()
        return render_template('rezultatai_cnn.html', 
                             cnn_results=cnn_results, 
                             cnn_modeliai=cnn_modeliai,
                             test_results=test_results)
    except Exception as e:
        return render_template('rezultatai_cnn.html', 
                             cnn_results=[], 
                             cnn_modeliai=[], 
                             test_results=[],
                             klaida=str(e))

@bp.route('/rezultatai_transformer')
def rezultatai_transformer():
    try:
        transformer_modeliai = []
        models_dir = 'app/static/models'
        if os.path.exists(models_dir):
            for fname in os.listdir(models_dir):
                if (fname.endswith('.joblib') or fname.endswith('.joblib.joblib') or fname.endswith('.pt')) and fname.startswith('transformer'):
                    transformer_modeliai.append({'pavadinimas': fname})
        transformer_results = []
        transformer_results_dir = os.path.join('app', 'results', 'transformer')
        eng2lt = {
            'accuracy': 'tikslumas',
            'precision': 'preciziškumas',
            'recall': 'atgaminimas',
            'f1': 'f1_balas',
        }
        if os.path.exists(transformer_results_dir):
            for filename in os.listdir(transformer_results_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(transformer_results_dir, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                    result['failo_pavadinimas'] = filename
                    # Metrikų raktų pervadinimas
                    if 'metrikos' in result and isinstance(result['metrikos'], dict):
                        for eng, lt in eng2lt.items():
                            if eng in result['metrikos']:
                                result['metrikos'][lt] = result['metrikos'][eng]
                    # Pridėti datą, jei jos nėra
                    if not result.get('data'):
                        import datetime
                        mtime = os.path.getmtime(file_path)
                        result['data'] = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                    # Pridėti duomenų failo pavadinimą, jei jo nėra
                    if not result.get('duomenu_failas'):
                        result['duomenu_failas'] = filename
                    transformer_results.append(result)
        query = TestResult.query.filter_by(modelio_tipas='transformer')
        test_results = query.order_by(TestResult.data.desc()).all()
        return render_template('rezultatai_transformer.html', 
                             transformer_results=transformer_results, 
                             transformer_modeliai=transformer_modeliai,
                             test_results=test_results)
    except Exception as e:
        return render_template('rezultatai_transformer.html', 
                             transformer_results=[], 
                             transformer_modeliai=[], 
                             test_results=[],
                             klaida=str(e)) 

@bp.route('/generuoti_grafikus', methods=['POST'])
def generuoti_grafikus():
    import json
    from app.utils.visualization import (
        sukurti_konfuzijos_macica,
        sukurti_klasiu_grafikus,
        sukurti_metriku_grafikus
    )
    modelio_tipas = request.form.get('modelio_tipas')
    result_file = request.form.get('result_file')
    if not result_file:
        return "Nepasirinktas rezultatų failas", 400

    # Nustatome pilną kelią
    file_path = os.path.join('app', 'results', result_file)
    if not os.path.exists(file_path):
        return "Failas nerastas", 404

    with open(file_path, encoding='utf-8') as f:
        data = json.load(f)

    # Universalus duomenų ištraukimas
    rez = data.get('rezultatai', data)
    if 'metrikos' in rez:
        metrikos = rez['metrikos']
    else:
        metrikos = rez

    y_true = rez.get('y_true')
    y_pred = rez.get('y_pred')
    klases = rez.get('klases')
    klasiu_tikslumas = rez.get('klasiu_tikslumas', [])
    # Metrikos
    tikslumas = float(metrikos.get('accuracy', metrikos.get('tikslumas', 0)))
    f1 = float(metrikos.get('f1', metrikos.get('f1_balas', 0)))
    precision = float(metrikos.get('precision', metrikos.get('preciziškumas', 0)))
    recall = float(metrikos.get('recall', metrikos.get('atgaminimas', 0)))

    # Katalogas pagal modelio tipą
    grafikai_dir = f'app/static/{modelio_tipas}_grafikai'
    os.makedirs(grafikai_dir, exist_ok=True)

    # Generuojame grafikus, jei yra duomenų
    if y_true and y_pred and klases:
        sukurti_konfuzijos_macica(
            y_true, y_pred, klases, modelio_tipas.upper(), f'{grafikai_dir}/konfuzijos_macica.png'
        )
    if klases and klasiu_tikslumas:
        sukurti_klasiu_grafikus(
            klases, klasiu_tikslumas, modelio_tipas.upper(), f'{grafikai_dir}/klasiu_grafikai.png'
        )
    sukurti_metriku_grafikus(
        {
            'Tikslumas': tikslumas,
            'F1 balas': f1,
            'Precision': precision,
            'Recall': recall
        },
        modelio_tipas.upper(),
        f'{grafikai_dir}/metriku_grafikai.png'
    )

    # Po generavimo grąžiname atgal į grafikų puslapį
    return redirect(url_for('main.grafikai', modelis=modelio_tipas)) 