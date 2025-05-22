import sys, os
import logging
import traceback
import codecs

# Nustatome logging su UTF-8 koduote
logging.basicConfig(
    filename='app/static/training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# Užtikriname, kad darbinis katalogas yra projekto šaknis
try:
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    logging.info("Sėkmingai nustatytas darbinis katalogas")
except Exception as e:
    logging.error(f"Klaida nustatant darbinį katalogą: {str(e)}")
    raise

import cv2
import numpy as np
import pandas as pd
from app.models.knn_model import KNNModelis
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time

GRAFIKU_KATALOGAS = 'app/static/knn_grafikai'
try:
    os.makedirs(GRAFIKU_KATALOGAS, exist_ok=True)
    os.makedirs('app/static', exist_ok=True)
    logging.info("Sukurti reikalingi katalogai")
except Exception as e:
    logging.error(f"Klaida kuriant katalogus: {str(e)}")
    raise

def apdoroti_paveiksleli(paveikslelio_kelias, roi=None):
    """
    Apdoroti paveikslėlį: nuskaityti, konvertuoti į pilkos spalvos, 
    iškirpti ROI ir normalizuoti
    """
    try:
        # Nuskaityti paveikslėlį
        img = cv2.imread(paveikslelio_kelias)
        if img is None:
            logging.error(f"Nepavyko nuskaityti paveikslėlio: {paveikslelio_kelias}")
            return None
        
        # Konvertuoti į pilkos spalvos
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Jei yra ROI, iškirpti
        if roi is not None:
            x1, y1, x2, y2 = roi
            img = img[y1:y2, x1:x2]
        
        # Normalizuoti dydį
        img = cv2.resize(img, (32, 32))
        
        # Užtikrinti, kad paveikslėlis būtų uint8 tipo
        img = img.astype(np.uint8)
        
        # Normalizuoti reikšmes į [0, 255] intervalą
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        
        return img
    except Exception as e:
        logging.error(f"Klaida apdorojant paveikslėlį {paveikslelio_kelias}: {str(e)}")
        return None

def paruosti_duomenis(train_csv_path, images_dir, augment=True):
    """
    Paruošti mokymo duomenis su galimybe augmentuoti
    """
    try:
        # Patikrinti ar failai egzistuoja
        if not os.path.exists(train_csv_path):
            raise FileNotFoundError(f"Duomenų failas nerastas: {train_csv_path}")
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Paveikslėlių katalogas nerastas: {images_dir}")

        # Nuskaityti mokymo duomenis
        train_data = pd.read_csv(train_csv_path)
        logging.info(f"Nuskaityti {len(train_data)} įrašų iš {train_csv_path}")
        
        X = []
        y = []
        X_original = []  # Išsaugoti originalius paveikslėlius vizualizacijai
        
        for idx, row in train_data.iterrows():
            try:
                image_path = os.path.join(images_dir, row['Path'])
                roi = (row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2'])
                
                img = apdoroti_paveiksleli(image_path, roi)
                if img is not None:
                    # Užtikrinti, kad paveikslėlis būtų uint8 tipo
                    img = img.astype(np.uint8)
                    
                    # Pridėti originalų paveikslėlį
                    X.append(img.flatten())
                    y.append(row['ClassId'])
                    X_original.append(img)
                    
                    # Jei reikia augmentuoti
                    if augment:
                        modelis = KNNModelis()
                        augmented = modelis.augment_paveiksleli(img)
                        for aug_img in augmented:
                            aug_img = aug_img.astype(np.uint8)
                            X.append(aug_img.flatten())
                            y.append(row['ClassId'])
            except Exception as e:
                logging.error(f"Klaida apdorojant {idx} įrašą: {str(e)}")
                continue
        
        # Konvertuoti į numpy masyvus
        X = np.array(X)
        y = np.array(y)
        X_original = np.array(X_original)
        
        # Užtikrinti, kad visi duomenys būtų teisingo tipo
        X = X.astype(np.float32)
        X_original = X_original.astype(np.uint8)
        
        logging.info(f"Paruošta {len(X)} pavyzdžių")
        return X, y, X_original
        
    except Exception as e:
        logging.error(f"Klaida paruošiant duomenis: {str(e)}")
        raise

def vizualizuoti_rezultatus(metrikos, cv_rezultatai, klaidu_analize):
    """
    Vizualizuoti modelio rezultatus
    """
    try:
        # Sukurti metrikų grafiką
        plt.figure(figsize=(12, 6))
        metrikos_values = [metrikos['tikslumas'], metrikos['preciziškumas'], 
                          metrikos['atgaminimas'], metrikos['f1_balas']]
        metrikos_names = ['Tikslumas', 'Preciziškumas', 'Atgaminimas', 'F1 balas']
        
        plt.bar(metrikos_names, metrikos_values)
        plt.title('Modelio Metrikos')
        plt.ylim(0, 1)
        plt.savefig(os.path.join(GRAFIKU_KATALOGAS, 'metrikos.png'))
        plt.close()
        
        # Sukurti cross-validation rezultatų grafiką
        plt.figure(figsize=(15, 8))
        metrikos = list(cv_rezultatai.keys())
        vidurkiai = [cv_rezultatai[m]['vidurkis'] for m in metrikos]
        std = [cv_rezultatai[m]['std'] for m in metrikos]
        
        # Rūšiuoti pagal vidurkį
        sorted_indices = np.argsort(vidurkiai)[::-1]
        metrikos = [metrikos[i] for i in sorted_indices]
        vidurkiai = [vidurkiai[i] for i in sorted_indices]
        std = [std[i] for i in sorted_indices]
        
        bars = plt.bar(range(len(metrikos)), vidurkiai, yerr=std, capsize=5)
        
        # Pridėti reikšmes virš stulpelių
        for bar, vidurkis in zip(bars, vidurkiai):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{vidurkis:.3f}', ha='center', va='bottom')
        
        plt.title('Kryžminės validacijos rezultatai')
        plt.ylabel('Tikslumas')
        plt.ylim(0, 1)
        plt.xticks(range(len(metrikos)), metrikos, rotation=45, ha='right')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(GRAFIKU_KATALOGAS, 'kryzmine_validacija.png'))
        plt.close()
        
        logging.info("Sėkmingai sugeneruoti grafikai")
        
    except Exception as e:
        logging.error(f"Klaida vizualizuojant rezultatus: {str(e)}")
        raise

def main():
    try:
        logging.info("Pradedamas modelio mokymas")
        
        # Patikrinti ar egzistuoja duomenų failai
        train_file = 'duomenys/knn_test.csv'
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Duomenų failas nerastas: {train_file}")
            
        modelis = KNNModelis()
        logging.info("Paruošiami duomenys...")
        X, y, X_original = paruosti_duomenis(train_file, 'duomenys', augment=True)
        logging.info(f"Paruošta {len(X)} pavyzdžių")
        
        # Mokyti modelį po 1000 pavyzdžių
        porcijos_dydis = 1000
        cv_rezultatai = {}
        klaidu_analize = {}
        
        for i in range(0, len(X), porcijos_dydis):
            porcijos_nr = i // porcijos_dydis + 1
            pabaiga = min(i + porcijos_dydis, len(X))
            logging.info(f"Mokoma su {pabaiga} pavyzdžių (iki {pabaiga})")
            
            try:
                # Mokyti modelį su dabartine porcija
                X_porcija = X[i:pabaiga]
                y_porcija = y[i:pabaiga]
                
                # Mokyti modelį
                metrikos = modelis.mokyti(X_porcija, y_porcija)
                
                # Atnaujinti progreso failą
                with codecs.open('app/static/progresas.json', 'w', encoding='utf-8') as f:
                    json.dump({
                        'status': 'vykdoma',
                        'progresas': f"{porcijos_nr}/{(len(X) + porcijos_dydis - 1) // porcijos_dydis}",
                        'paskutinis_atnaujinimas': time.strftime('%Y-%m-%d %H:%M:%S')
                    }, f, ensure_ascii=False)
                
                # Išsaugoti rezultatus
                rezultatu_failas = f'app/static/knn_porciju_rezultatai.json'
                try:
                    with codecs.open(rezultatu_failas, 'w', encoding='utf-8') as f:
                        json.dump({
                            'porcijos_nr': porcijos_nr,
                            'metrikos': metrikos,
                            'pavyzdziu_skaicius': len(X_porcija)
                        }, f, ensure_ascii=False, indent=4)
                except Exception as e:
                    logging.error(f"Klaida išsaugant rezultatus: {str(e)}")
                
            except Exception as e:
                logging.error(f"Klaida apdorojant {porcijos_nr} porciją: {str(e)}")
                continue
        
        # Vizualizuoti galutinius rezultatus
        try:
            vizualizuoti_rezultatus(metrikos, cv_rezultatai, klaidu_analize)
        except Exception as e:
            logging.error(f"Klaida vizualizuojant galutinius rezultatus: {str(e)}")
        
        # Atnaujinti progreso failą
        with codecs.open('app/static/progresas.json', 'w', encoding='utf-8') as f:
            json.dump({
                'status': 'baigta',
                'progresas': '100%',
                'paskutinis_atnaujinimas': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, ensure_ascii=False)
        
        logging.info("Modelio mokymas sėkmingai baigtas")
        
    except Exception as e:
        logging.error(f"Kritinė klaida: {str(e)}\n{traceback.format_exc()}")
        # Atnaujinti progreso failą su klaida
        try:
            with codecs.open('app/static/progresas.json', 'w', encoding='utf-8') as f:
                json.dump({
                    'status': 'klaida',
                    'klaidos_zinute': str(e),
                    'paskutinis_atnaujinimas': time.strftime('%Y-%m-%d %H:%M:%S')
                }, f, ensure_ascii=False)
        except Exception as write_error:
            logging.error(f"Klaida rašant į progreso failą: {str(write_error)}")
        raise

if __name__ == '__main__':
    main() 