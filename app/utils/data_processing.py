import os
import sys
import codecs
import json
import logging
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess_image(image_path, target_size=(32, 32)):
    """
    Užkrauti ir apdoroti vieną paveikslėlį
    """
    try:
        # Nuskaityti paveikslėlį
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Nepavyko nuskaityti paveikslėlio: {image_path}")
        
        # Konvertuoti į RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Pakeisti dydį
        img = cv2.resize(img, target_size)
        
        # Normalizuoti
        img = img.astype(np.float32) / 255.0
        
        return img
    except FileNotFoundError as e:
        print(f'Failas nerastas: {str(e)}')
        raise
    except ValueError as e:
        print(f'Neteisingi duomenys: {str(e)}')
        raise
    except Exception as e:
        print(f'Įvyko nenumatyta klaida: {str(e)}')
        raise

def extract_hog_features(image):
    """
    Ištraukti HOG požymius iš paveikslėlio
    """
    # Jei float, konvertuoti į uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    # Konvertuoti į pilką
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Skaičiuoti HOG požymius
    win_size = (32, 32)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    features = hog.compute(gray)
    
    return features.flatten()

def extract_haar_features(image):
    """
    Ištraukti Haar požymius iš paveikslėlio
    """
    # Jei float, konvertuoti į uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    # Konvertuoti į pilką
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Sukurti Haar kaskadą
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Aptikti požymius
    features = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Konvertuoti į fiksuoto dydžio vektorių
    if len(features) > 0:
        features = features.flatten()
    else:
        features = np.zeros(4)  # Jei nėra požymių
    
    return features

def extract_hue_histogram(image, bins=32):
    """
    Extract hue histogram from an image
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Calculate histogram
    hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    
    # Normalize histogram
    hist = cv2.normalize(hist, hist).flatten()
    
    return hist

def load_dataset_cnn(data_dir, annotations_file):
    print('KVIEČIAMA: load_dataset_cnn')
    """
    Užkrauti GTSRB duomenų rinkinį su diagnostika ir papildomu patikrinimu (CNN)
    """
    # Nuskaityti anotacijas
    try:
        annotations = pd.read_csv(annotations_file)
    except Exception as e:
        logging.error(f'Nepavyko nuskaityti CSV failo: {annotations_file}. Klaida: {str(e)}')
        raise ValueError(f'Nepavyko nuskaityti CSV failo: {annotations_file}. Klaida: {str(e)}')

    # Patikrinti ar yra reikiami stulpeliai
    if not (('Path' in annotations.columns or 'Filename' in annotations.columns) and 'ClassId' in annotations.columns):
        logging.error(f'CSV faile turi būti stulpeliai "Path" arba "Filename" ir "ClassId". Rasta: {annotations.columns}')
        raise ValueError(f'CSV faile turi būti stulpeliai "Path" arba "Filename" ir "ClassId". Rasta: {annotations.columns}')

    images = []
    labels = []
    missing = 0
    total = 0
    for idx, row in annotations.iterrows():
        total += 1
        img_col = 'Path' if 'Path' in row else 'Filename'
        image_path = os.path.join(data_dir, row[img_col])
        if not os.path.exists(image_path):
            logging.warning(f'Nerastas paveikslėlis: {image_path}')
            missing += 1
            continue
        img = cv2.imread(image_path)
        if img is None:
            logging.warning(f'Nepavyko nuskaityti paveikslėlio: {image_path}')
            missing += 1
            continue
        img = cv2.resize(img, (32, 32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        images.append(img)
        labels.append(row['ClassId'])
    if missing > 0:
        logging.warning(f'Iš viso praleista {missing} paveikslėlių iš {total} (nerasta arba nepavyko nuskaityti)')
    if len(images) == 0:
        raise ValueError('Nepavyko užkrauti nė vieno paveikslėlio. Patikrinkite CSV ir paveikslėlių katalogą.')
    logging.info(f'Sėkmingai užkrauta {len(images)} paveikslėlių iš {total}')
    print('load_dataset_cnn grąžina:', type(images), len(images), type(labels), len(labels))
    print('labels pavyzdžiai:', labels[:10])
    # Papildomas patikrinimas
    labels_arr = np.array(labels)
    if not np.issubdtype(labels_arr.dtype, np.integer):
        raise ValueError(f'labels turi būti sveikųjų skaičių masyvas, bet gauta: {labels_arr.dtype}, pavyzdžiai: {labels_arr[:10]}')
    return np.array(images), labels_arr

def prepare_data_for_training(images, labels, test_size=0.2, val_size=0.1):
    """
    Paruošti duomenis mokymui, validacijai ir testavimui
    """
    # Pirmiausia padalinti į mokymo+validacijos ir testavimo aibes
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Tada padalinti mokymo+validacijos į mokymo ir validacijos aibes
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size/(1-test_size),
        random_state=42, stratify=y_train_val
    )
    
    # Konvertuoti žymes į one-hot kodavimą
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.transform(y_val)
    y_test = label_encoder.transform(y_test)
    
    y_train = np.eye(len(label_encoder.classes_))[y_train]
    y_val = np.eye(len(label_encoder.classes_))[y_val]
    y_test = np.eye(len(label_encoder.classes_))[y_test]
    
    return X_train, y_train, X_val, y_val, X_test, y_test, label_encoder

def process_image(image_path, target_size=(32, 32)):
    """
    Apdoroti paveikslėlį
    """
    try:
        # Nuskaityti paveikslėlį
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Nepavyko nuskaityti paveikslėlio: {image_path}")
            
        # Pakeisti dydį
        image = cv2.resize(image, target_size)
        
        # Konvertuoti į RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalizuoti
        image = image.astype(np.float32) / 255.0
        
        return image
        
    except Exception as e:
        logging.error(f"Klaida apdorojant paveikslėlį: {str(e)}")
        raise
        
def augment_image(image):
    """
    Atlikti paveikslėlio augmentaciją
    """
    try:
        augmented = []
        
        # Rotacija
        for angle in [-15, 15]:
            M = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), angle, 1)
            rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            augmented.append(rotated)
            
        # Ryškumo keitimas
        bright = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
        dark = cv2.convertScaleAbs(image, alpha=0.8, beta=-10)
        augmented.extend([bright, dark])
        
        # Horizontali apvertimas
        flipped = cv2.flip(image, 1)
        augmented.append(flipped)
        
        # Triukšmo pridėjimas
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        noisy = cv2.add(image, noise)
        augmented.append(noisy)
        
        # Kontrasto keitimas
        contrast = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
        augmented.append(contrast)
        
        return augmented
        
    except Exception as e:
        logging.error(f"Klaida augmentuojant paveikslėlį: {str(e)}")
        raise
        
def prepare_data(X, y, augment=False, model_type='knn'):
    """
    Paruošti duomenis
    """
    try:
        if model_type == 'cnn':
            # CNN modeliui palikti 4D formą (batch_size, height, width, channels)
            X_scaled = X.astype(np.float32) / 255.0
        else:
            # Kitiems modeliams suformuoti 2D
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X.reshape(X.shape[0], -1))
        
        if augment:
            # Atlikti augmentaciją
            X_augmented = []
            y_augmented = []
            
            for i in range(len(X)):
                augmented = augment_image(X[i])
                X_augmented.extend(augmented)
                y_augmented.extend([y[i]] * len(augmented))
                
            X_augmented = np.array(X_augmented)
            y_augmented = np.array(y_augmented)
            
            if model_type == 'cnn':
                # CNN modeliui palikti 4D formą
                X_augmented_scaled = X_augmented.astype(np.float32) / 255.0
            else:
                # Kitiems modeliams suformuoti 2D
                X_augmented_scaled = scaler.transform(X_augmented.reshape(X_augmented.shape[0], -1))
            
            return X_augmented_scaled, y_augmented
            
        return X_scaled, y
        
    except Exception as e:
        logging.error(f"Klaida ruošiant duomenis: {str(e)}")
        raise
        
def save_data(X, y, path):
    """
    Išsaugoti duomenis
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data = {
            'X': X.tolist(),
            'y': y.tolist()
        }
        
        with codecs.open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
    except Exception as e:
        logging.error(f"Klaida išsaugant duomenis: {str(e)}")
        raise
        
def load_data(path):
    """
    Užkrauti duomenis
    """
    try:
        with codecs.open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        X = np.array(data['X'])
        y = np.array(data['y'])
        
        return X, y
        
    except Exception as e:
        logging.error(f"Klaida užkraunant duomenis: {str(e)}")
        raise

def universalus_duomenu_nuskaitymas(csv_path, base_dir='duomenys', img_size=32):
    print('KVIEČIAMA: universalus_duomenu_nuskaitymas')
    import pandas as pd
    import numpy as np
    import cv2
    df = pd.read_csv(csv_path)
    columns = set(df.columns)
    # Atvejis A: CSV su paveikslėlių keliais
    if 'Path' in columns and 'ClassId' in columns:
        X = []
        y = []
        for idx, row in df.iterrows():
            img_path = os.path.join(base_dir, row['Path'])
            class_id = row['ClassId']
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Nepavyko nuskaityti: {img_path}")
                continue
            img = cv2.resize(img, (img_size, img_size))
            X.append(img.flatten())
            y.append(class_id)
        X = np.array(X)
        y = np.array(y)
        return X, y
    # Atvejis B: CSV jau su flatten duomenimis
    elif 'class_id' in columns:
        X = df.drop(['class_id', 'class_name', 'file_name'], axis=1).values
        y = df['class_id'].values
        return X, y
    else:
        raise ValueError("CSV faile nerasta tinkamų stulpelių (turi būti 'Path' ir 'ClassId' ARBA 'class_id').") 