import os
import sys
import codecs
import json
import logging
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from app.utils.visualization import (
    sukurti_konfuzijos_macica,
    sukurti_klasiu_grafikus,
    sukurti_metriku_grafikus,
    issaugoti_rezultatus
)
import joblib
from app.model_progress import model_progress
import time

class KNNModelis:
    """
    KNN modelio klasė
    """
    def __init__(self):
        self.model = None
        self.best_k = None
        self.best_metric = None
        self.logger = logging.getLogger('app')
        
    def mokyti(self, X, y, mokymo_tipas=None):
        """
        Mokyti modelį
        """
        try:
            start_time = time.time()
            # Padalinti duomenis į mokymo ir testavimo aibes
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )
            model_progress.start('knn', len(X_train))
            # Rasti geriausią k reikšmę su progresu
            k_values = range(1, 21)
            cv_scores = []
            for idx, k in enumerate(k_values):
                if model_progress.should_stop('knn'):
                    model_progress.error('knn', 'Training stopped by user')
                    self._save_partial_model()
                    return None
                model = KNeighborsClassifier(n_neighbors=k)
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                cv_scores.append(scores.mean())
                elapsed = time.time() - start_time
                percent = int((idx+1)/len(k_values)*10)
                eta = self._estimate_eta(start_time, idx+1, len(k_values))
                model_progress.update('knn', percent, message=f'CV: k={k}, {idx+1}/{len(k_values)}, ETA: {eta}s')
            self.best_k = k_values[np.argmax(cv_scores)]
            self.best_metric = max(cv_scores)
            self.logger.info(f"Geriausia k reikšmė: {self.best_k} (tikslumas: {self.best_metric:.4f})")
            if model_progress.should_stop('knn'):
                model_progress.error('knn', 'Training stopped by user')
                self._save_partial_model()
                return None
            self.model = KNeighborsClassifier(
                n_neighbors=self.best_k,
                weights='uniform',
                algorithm='auto',
                leaf_size=30,
                p=2,
                metric='minkowski',
                n_jobs=-1
            )
            # Fit su progreso atnaujinimu (imitacija, nes sklearn fit neturi callback)
            self.model.fit(X_train, y_train)
            for i in range(1, len(X_train)+1, max(1, len(X_train)//10)):
                if model_progress.should_stop('knn'):
                    model_progress.error('knn', 'Training stopped by user')
                    self._save_partial_model()
                    return None
                elapsed = time.time() - start_time
                percent = int(i/len(X_train)*100)
                eta = self._estimate_eta(start_time, i, len(X_train))
                model_progress.update('knn', percent, message=f"Mokoma KNN: {i}/{len(X_train)}, ETA: {eta}s")
            model_progress.finish('knn')
            # Gauti prognozes
            y_pred = self.model.predict(X_test)
            # Apskaičiuoti metrikas
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            # Apskaičiuoti klasių tikslumą
            klasiu_tikslumas = []
            for klase in np.unique(y):
                mask = y_test == klase
                klasiu_tikslumas.append(accuracy_score(y_test[mask], y_pred[mask]))
            # Sukurti grafikus
            sukurti_konfuzijos_macica(
                y_test,
                y_pred,
                np.unique(y),
                'KNN',
                'app/static/knn_grafikai/konfuzijos_macica.png'
            )
            sukurti_klasiu_grafikus(
                np.unique(y),
                klasiu_tikslumas,
                'KNN',
                'app/static/knn_grafikai/klasiu_grafikai.png'
            )
            sukurti_metriku_grafikus(
                {
                    'Tikslumas': float(accuracy),
                    'F1 balas': float(f1),
                    'Precision': float(precision),
                    'Recall': float(recall)
                },
                'KNN',
                'app/static/knn_grafikai/metriku_grafikai.png'
            )
            # Išsaugoti rezultatus
            results = {
                'y_true': y_test.tolist(),
                'y_pred': y_pred.tolist(),
                'klases': np.unique(y).tolist(),
                'klasiu_tikslumas': klasiu_tikslumas,
                'metrikos': {
                    'accuracy': accuracy,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall
                }
            }
            issaugoti_rezultatus(
                results,
                'app/static/knn_porciju_rezultatai.json'
            )
            self.logger.info(f"KNN modelis sėkmingai išmokytas (k={self.best_k})")
            self.issaugoti_modeli('app/static/models/knn_model')
            return results
        except Exception as e:
            self.logger.error(f"Klaida mokant KNN modelį: {str(e)}")
            raise

    def _save_partial_model(self):
        try:
            if self.model is not None:
                joblib.dump(self.model, 'app/static/models/knn_model_partial.joblib')
                self.logger.info('Išsaugotas dalinis KNN modelis (nutraukus mokymą)')
        except Exception as e:
            self.logger.error(f'Klaida išsaugant dalinį modelį: {str(e)}')

    def _estimate_eta(self, start_time, current, total):
        elapsed = time.time() - start_time
        if current == 0:
            return None
        avg_per_item = elapsed / current
        remaining = total - current
        eta = int(avg_per_item * remaining)
        return eta

    def prognozuoti(self, X):
        """
        Prognozuoti klasę ir tikimybę
        """
        try:
            if self.model is None:
                raise ValueError("Modelis nėra išmokytas")
            pred = self.model.predict(X.reshape(1, -1))[0]
            proba = self.model.predict_proba(X.reshape(1, -1))[0]
            confidence = float(np.max(proba))
            return int(pred), confidence
        except Exception as e:
            self.logger.error(f"Klaida prognozuojant su KNN modeliu: {str(e)}")
            raise
            
    def issaugoti_modeli(self, kelias):
        """
        Išsaugoti modelį
        """
        try:
            if self.model is None:
                raise ValueError("Modelis nėra išmokytas")

            os.makedirs(os.path.dirname(kelias), exist_ok=True)

            # Jei kelias jau baigiasi .joblib, nepridėti dar kartą
            if kelias.endswith('.joblib'):
                modelio_failas = kelias
                json_failas = kelias[:-7] + '.json'  # pašalinam .joblib
            else:
                modelio_failas = kelias + '.joblib'
                json_failas = kelias + '.json'

            # Išsaugome visą modelį su joblib
            joblib.dump(self.model, modelio_failas)

            # Papildomai išsaugome parametrus į JSON
            model_data = {
                'k': self.best_k,
                'best_metric': self.best_metric
            }
            with codecs.open(json_failas, 'w', encoding='utf-8') as f:
                json.dump(model_data, f, ensure_ascii=False, indent=4)

            self.logger.info(f"KNN modelis sėkmingai išsaugotas: {modelio_failas} ir {json_failas}")

        except Exception as e:
            self.logger.error(f"Klaida išsaugant KNN modelį: {str(e)}")
            raise
            
    def uzsikrauti_modeli(self, kelias):
        """
        Užsikrauti modelį
        """
        try:
            with codecs.open(kelias + '.json', 'r', encoding='utf-8') as f:
                model_data = json.load(f)
                
            self.best_k = model_data['k']
            self.best_metric = model_data['best_metric']
            
            self.model = KNeighborsClassifier(
                n_neighbors=self.best_k,
                weights='uniform',
                algorithm='auto',
                leaf_size=30,
                p=2,
                metric='minkowski',
                n_jobs=-1
            )
            
            self.logger.info(f"KNN modelis sėkmingai užkrautas: {kelias}")
            
        except Exception as e:
            self.logger.error(f"Klaida užkraunant KNN modelį: {str(e)}")
            raise 