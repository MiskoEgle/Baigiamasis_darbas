import os
import sys
import codecs
import json
import logging
import numpy as np
import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from app.utils.data_processing import process_image, augment_image

def load_dataset(data_dir, test_size=0.2, val_size=0.1, augment=False):
    """
    Užkrauti duomenų rinkinį
    """
    try:
        # Užkrauti duomenis
        X = []
        y = []
        
        for class_dir in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
                
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                if not image_file.endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                # Apdoroti paveikslėlį
                image = process_image(image_path)
                if image is not None:
                    X.append(image)
                    y.append(class_dir)
                    
                    # Atlikti augmentaciją
                    if augment:
                        augmented = augment_image(image)
                        X.extend(augmented)
                        y.extend([class_dir] * len(augmented))
        
        # Konvertuoti į numpy masyvus
        X = np.array(X)
        y = np.array(y)
        
        # Koduoti etiketes
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        
        # Padalinti duomenis
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=val_size,
            random_state=42,
            stratify=y_train
        )
        
        return X_train, y_train, X_val, y_val, X_test, y_test, label_encoder
        
    except Exception as e:
        logging.error(f"Klaida užkraunant duomenų rinkinį: {str(e)}")
        raise
        
def save_dataset_info(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder, path):
    """
    Išsaugoti duomenų rinkinio informaciją
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        info = {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'num_classes': len(label_encoder.classes_),
            'classes': label_encoder.classes_.tolist(),
            'class_distribution': {
                'train': np.bincount(y_train).tolist(),
                'val': np.bincount(y_val).tolist(),
                'test': np.bincount(y_test).tolist()
            }
        }
        
        with codecs.open(path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=4)
            
    except Exception as e:
        logging.error(f"Klaida išsaugant duomenų rinkinio informaciją: {str(e)}")
        raise
        
def load_dataset_info(path):
    """
    Užkrauti duomenų rinkinio informaciją
    """
    try:
        with codecs.open(path, 'r', encoding='utf-8') as f:
            info = json.load(f)
            
        return info
        
    except Exception as e:
        logging.error(f"Klaida užkraunant duomenų rinkinio informaciją: {str(e)}")
        raise

def nuskaityti_is_zip(zip_path):
    """
    Nuskaityti duomenis iš ZIP failo
    """
    try:
        # Patikrinti ar failas egzistuoja
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"ZIP failas nerastas: {zip_path}")
            
        # Sukurti laikiną katalogą
        temp_dir = 'temp_data'
        os.makedirs(temp_dir, exist_ok=True)
        
        # Išskleisti ZIP failą
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
        # Nuskaityti CSV failą
        csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("CSV failas nerastas ZIP faile")
            
        csv_path = os.path.join(temp_dir, csv_files[0])
        data = pd.read_csv(csv_path)
        
        # Apdoroti paveikslėlius
        X = []
        y = []
        
        for idx, row in data.iterrows():
            try:
                image_path = os.path.join(temp_dir, row['Path'])
                img = process_image(image_path)
                if img is not None:
                    X.append(img.flatten())
                    y.append(row['ClassId'])
            except Exception as e:
                logging.error(f"Klaida apdorojant {idx} įrašą: {str(e)}")
                continue
                
        # Konvertuoti į numpy masyvus
        X = np.array(X)
        y = np.array(y)
        
        # Išvalyti laikiną katalogą
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)
        
        return X, y
        
    except Exception as e:
        logging.error(f"Klaida nuskaitant duomenis iš ZIP: {str(e)}")
        raise

def nuskaityti_is_csv(csv_path):
    """
    Nuskaityti duomenis iš CSV failo
    """
    try:
        # Patikrinti ar failas egzistuoja
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV failas nerastas: {csv_path}")
            
        # Nuskaityti CSV failą
        data = pd.read_csv(csv_path)
        
        # Apdoroti paveikslėlius
        X = []
        y = []
        
        for idx, row in data.iterrows():
            try:
                image_path = os.path.join(os.path.dirname(csv_path), row['Path'])
                img = process_image(image_path)
                if img is not None:
                    X.append(img.flatten())
                    y.append(row['ClassId'])
            except Exception as e:
                logging.error(f"Klaida apdorojant {idx} įrašą: {str(e)}")
                continue
                
        # Konvertuoti į numpy masyvus
        X = np.array(X)
        y = np.array(y)
        
        return X, y
        
    except Exception as e:
        logging.error(f"Klaida nuskaitant duomenis iš CSV: {str(e)}")
        raise 