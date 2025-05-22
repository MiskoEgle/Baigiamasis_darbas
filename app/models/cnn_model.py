import os
import sys
import json
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from app.utils.visualization import (
    sukurti_konfuzijos_macica,
    sukurti_klasiu_grafikus,
    sukurti_metriku_grafikus,
    issaugoti_rezultatus
)
from sklearn.model_selection import train_test_split
from app.models.database import db, CNNModelDB
import joblib
import io
from app.model_progress import model_progress
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Nustatyti UTF-8 koduotę
# sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
# sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

class CNNModelis:
    """
    CNN modelio klasė
    """
    def __init__(self):
        self.model = None
        self.history = None
        self.logger = logging.getLogger('app')
        self.ivesties_forma = (32, 32, 3)  # Numatytasis įvesties dydis GTSRB
        self.klasiu_skaicius = 43  # Kelio ženklų klasių skaičius GTSRB
        
    def mokyti(self, X, y, mokymo_tipas='naujas'):
        """
        Mokyti modelį
        """
        try:
            # Užtikrinti, kad y būtų one-hot encoded ir float32
            print('Prieš LabelEncoder: y dtype:', y.dtype, 'min:', y.min(), 'max:', y.max())
            le = LabelEncoder()
            y = le.fit_transform(y)
            y = tf.keras.utils.to_categorical(y, num_classes=self.klasiu_skaicius)
            y = y.astype(np.float32)  # Konvertuoti į float32
            print('Po one-hot encoding: y shape:', y.shape, 'dtype:', y.dtype)
            
            # Užtikrinti, kad X būtų float32
            X = X.astype(np.float32)
            
            # Padalinti duomenis į mokymo ir testavimo aibes
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=np.argmax(y, axis=1)  # Use class indices for stratification
            )
            
            print('X_train shape:', X_train.shape, 'dtype:', X_train.dtype)
            print('y_train shape:', y_train.shape, 'dtype:', y_train.dtype)
            
            n_epochs = 50
            batch_size = 16
            model_progress.start('cnn', n_epochs)
            
            # --- Nauja logika: tęsti ar naujas mokymas ---
            model_path = 'app/static/models/cnn_model'
            if mokymo_tipas == 'testi' and os.path.exists(model_path + '.h5'):
                print('Tęsiu mokymą: įkeliu esamą modelį')
                self.uzsikrauti_modeli(model_path)
            else:
                print('Naujas mokymas: kuriu naują modelį')
                self.model = self._sukurti_modeli(X.shape[1:], self.klasiu_skaicius)
            
            # Custom training loop
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
            loss_fn = tf.keras.losses.CategoricalCrossentropy()
            
            # Create TensorFlow datasets
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
            
            # Training loop
            for epoch in range(n_epochs):
                print(f"\nEpoch {epoch + 1}/{n_epochs}")
                total_loss = 0
                num_batches = 0
                
                for batch_idx, (x_batch, y_batch) in enumerate(train_dataset):
                    with tf.GradientTape() as tape:
                        # Forward pass
                        predictions = self.model(x_batch, training=True)
                        loss = loss_fn(y_batch, predictions)
                    # Backward pass
                    gradients = tape.gradient(loss, self.model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                    total_loss += loss
                    num_batches += 1
                    if batch_idx % 10 == 0:
                        print(f"Batch {batch_idx}, Loss: {loss:.4f}")
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
                if model_progress.should_stop('cnn'):
                    model_progress.error('cnn', 'Training stopped by user')
                    return None
                model_progress.update('cnn', epoch + 1)
            model_progress.finish('cnn')

            # Išsaugoti modelį iškart po mokymo
            self.issaugoti_modeli(model_path)
            print('Modelis išsaugotas: app/static/models/cnn_model.h5')

            # Gauti prognozes
            y_pred = self.model.predict(X_test)
            y_pred = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_test, axis=1)
            
            # Apskaičiuoti metrikas su zero_division=0
            accuracy = float(accuracy_score(y_true, y_pred))
            f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
            precision = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
            recall = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
            
            print(f"Metrics - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            
            # Apskaičiuoti klasių tikslumą
            klasiu_tikslumas = []
            for klase in range(self.klasiu_skaicius):
                mask = y_true == klase
                if np.any(mask):  # Check if class exists in test set
                    klasiu_tikslumas.append(float(accuracy_score(y_true[mask], y_pred[mask])))
                else:
                    klasiu_tikslumas.append(0.0)
            
            # Sukurti grafikus
            try:
                sukurti_konfuzijos_macica(
                    y_true,
                    y_pred,
                    range(self.klasiu_skaicius),
                    'CNN',
                    'app/static/cnn_grafikai/konfuzijos_macica.png'
                )
                sukurti_klasiu_grafikus(
                    range(self.klasiu_skaicius),
                    klasiu_tikslumas,
                    'CNN',
                    'app/static/cnn_grafikai/klasiu_grafikai.png'
                )
                sukurti_metriku_grafikus(
                    {
                        'Tikslumas': [float(accuracy)],
                        'F1 balas': [float(f1)],
                        'Precision': [float(precision)],
                        'Recall': [float(recall)]
                    },
                    'CNN',
                    'app/static/cnn_grafikai/metriku_grafikai.png'
                )
            except Exception as e:
                print(f"Warning: Error creating visualizations: {str(e)}")
            
            # Išsaugoti rezultatus su unikaliu pavadinimu
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results = {
                'y_true': y_true.tolist(),
                'y_pred': y_pred.tolist(),
                'klases': list(range(self.klasiu_skaicius)),
                'klasiu_tikslumas': [float(x) for x in klasiu_tikslumas],
                'metrikos': {
                    'accuracy': float(accuracy),
                    'f1': float(f1),
                    'precision': float(precision),
                    'recall': float(recall)
                },
                'mokymo_tipas': mokymo_tipas,
                'data': timestamp
            }
            results_path = f'app/results/cnn/cnn_rezultatai_{timestamp}.json'
            issaugoti_rezultatus(
                results,
                results_path
            )
            self.logger.info(f"CNN modelis sėkmingai išmokytas, rezultatai: {results_path}")
            return results
        except Exception as e:
            self.logger.error(f"Klaida mokant CNN modelį: {str(e)}")
            raise
            
    def prognozuoti(self, X):
        """
        Prognozuoti klasę ir grąžinti prognozę bei tikimybę
        """
        try:
            if self.model is None:
                raise ValueError("Modelis nėra išmokytas")
            X = X.reshape(1, *X.shape)
            y_pred = self.model.predict(X)
            tikimybes = y_pred[0]
            prognoze = int(np.argmax(tikimybes))
            tikimybe = float(np.max(tikimybes))
            return prognoze, tikimybe
        except Exception as e:
            self.logger.error(f"Klaida prognozuojant su CNN modeliu: {str(e)}")
            raise
            
    def _sukurti_modeli(self, input_shape, num_classes):
        """
        Sukurti CNN modelį
        """
        try:
            # Jei input_shape yra 2D, konvertuoti į 4D
            if len(input_shape) == 2:
                input_shape = (32, 32, 3)  # Numatytasis GTSRB dydis
            
            # Naudoti paprastesnį modelį
            model = models.Sequential([
                layers.Input(shape=input_shape),
                layers.Conv2D(16, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(32, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(32, activation='relu'),
                layers.Dense(num_classes, activation='softmax')
            ])
            
            # Naudoti SGD optimizer
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
            
            model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"Klaida kuriant CNN modelį: {str(e)}")
            raise
            
    def issaugoti_modeli(self, kelias):
        """
        Išsaugoti modelį
        """
        try:
            if self.model is None:
                raise ValueError("Modelis nėra išmokytas")
            self.model.save(kelias + '.h5')
            self.logger.info(f"CNN modelis sėkmingai išsaugotas: {kelias}")
        except Exception as e:
            self.logger.error(f"Klaida išsaugant CNN modelį: {str(e)}")
            raise
            
    def uzsikrauti_modeli(self, kelias):
        """
        Užsikrauti modelį
        """
        try:
            from tensorflow.keras import models
            if not kelias.endswith('.h5'):
                kelias = kelias + '.h5'
            self.model = models.load_model(kelias)
            self.logger.info(f"CNN modelis sėkmingai užkrautas: {kelias}")
        except Exception as e:
            self.logger.error(f"Klaida užkraunant CNN modelį: {str(e)}")
            raise

    def issaugoti_i_db(self, pavadinimas=None):
        if self.model is None:
            raise ValueError("Nėra modelio išsaugojimui")
        # Serializuoti modelį į baitus
        modelio_duomenys = {
            'modelis': self.model,
            'skaleris': self.skaleris,
            'klases': self.klases
        }
        buffer = io.BytesIO()
        joblib.dump(modelio_duomenys, buffer)
        buffer.seek(0)
        db_model = CNNModelDB(modelio_failas=buffer.read(), pavadinimas=pavadinimas)
        db.session.add(db_model)
        db.session.commit()

    def ikelti_is_db(self):
        # Paimti naujausią modelį iš DB
        db_model = CNNModelDB.query.order_by(CNNModelDB.data.desc()).first()
        if db_model is None:
            raise FileNotFoundError("Modelis DB nerastas")
        buffer = io.BytesIO(db_model.modelio_failas)
        modelio_duomenys = joblib.load(buffer)
        self.model = modelio_duomenys['modelis']
        self.skaleris = modelio_duomenys['skaleris']
        self.klases = modelio_duomenys.get('klases') 