import os
import sys
import codecs
import json
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive 'Agg'
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

def nustatyti_stiliu():
    """
    Nustatyti grafikų stilių
    """
    plt.style.use('seaborn')
    sns.set_palette("husl")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

def sukurti_konfuzijos_macica(y_true, y_pred, klases, modelio_tipas, kelias):
    """
    Sukurti konfuzijos matricą
    """
    try:
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{modelio_tipas} Konfuzijos Matrica')
        plt.ylabel('Tikroji klasė')
        plt.xlabel('Prognozuota klasė')
        
        os.makedirs(os.path.dirname(kelias), exist_ok=True)
        plt.savefig(kelias)
        plt.close()
        
    except Exception as e:
        logging.error(f"Klaida kuriant konfuzijos matricą: {str(e)}")
        raise

def sukurti_klasiu_grafikus(klases, tikslumas, modelio_tipas, kelias):
    """
    Sukurti klasių tikslumo grafikus
    """
    try:
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(klases)), tikslumas)
        plt.title(f'{modelio_tipas} Klasių Tikslumas')
        plt.xlabel('Klasė')
        plt.ylabel('Tikslumas')
        plt.xticks(range(len(klases)), klases, rotation=45)
        
        os.makedirs(os.path.dirname(kelias), exist_ok=True)
        plt.savefig(kelias)
        plt.close()
        
    except Exception as e:
        logging.error(f"Klaida kuriant klasių grafikus: {str(e)}")
        raise

def sukurti_metriku_grafikus(metrikos, modelio_tipas, kelias):
    """
    Sukurti metrikų grafikus
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.bar(metrikos.keys(), metrikos.values())
        plt.title(f'{modelio_tipas} Metrikos')
        plt.xlabel('Metrika')
        plt.ylabel('Reikšmė')
        plt.xticks(rotation=45)
        
        os.makedirs(os.path.dirname(kelias), exist_ok=True)
        plt.savefig(kelias)
        plt.close()
        
    except Exception as e:
        logging.error(f"Klaida kuriant metrikų grafikus: {str(e)}")
        raise

def issaugoti_rezultatus(rezultatai, kelias):
    """
    Išsaugoti rezultatus
    """
    try:
        os.makedirs(os.path.dirname(kelias), exist_ok=True)
        
        with codecs.open(kelias, 'w', encoding='utf-8') as f:
            json.dump(rezultatai, f, ensure_ascii=False, indent=4)
            
    except Exception as e:
        logging.error(f"Klaida išsaugant rezultatus: {str(e)}")
        raise

def plot_training_history(history, save_path=None):
    """
    Nubraižyti mokymo ir validacijos metrikas
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Nubraižyti tikslumą
    ax1.plot(history.history['accuracy'], label='Mokymo tikslumas')
    if 'val_accuracy' in history.history:
        ax1.plot(history.history['val_accuracy'], label='Validacijos tikslumas')
    ax1.set_title('Modelio tikslumas')
    ax1.set_xlabel('Epoka')
    ax1.set_ylabel('Tikslumas')
    ax1.legend()
    
    # Nubraižyti nuostolius
    ax2.plot(history.history['loss'], label='Mokymo nuostoliai')
    if 'val_loss' in history.history:
        ax2.plot(history.history['val_loss'], label='Validacijos nuostoliai')
    ax2.set_title('Modelio nuostoliai')
    ax2.set_xlabel('Epoka')
    ax2.set_ylabel('Nuostoliai')
    ax2.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Nubraižyti sumaišties (confusion) matricą
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Klaidų matrica')
    plt.xlabel('Prognozuota')
    plt.ylabel('Tikroji')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_class_distribution(labels, class_names, save_path=None):
    """
    Nubraižyti klasių pasiskirstymą duomenų rinkinyje
    """
    plt.figure(figsize=(15, 5))
    sns.countplot(x=labels)
    plt.title('Kelių ženklų klasių pasiskirstymas')
    plt.xlabel('Klasė')
    plt.ylabel('Kiekis')
    plt.xticks(rotation=45)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_feature_importance(feature_importance, feature_names, save_path=None):
    """
    Nubraižyti požymių svarbą
    """
    plt.figure(figsize=(10, 6))
    importance_df = pd.DataFrame({
        'Požymis': feature_names,
        'Svarba': feature_importance
    })
    importance_df = importance_df.sort_values('Svarba', ascending=False)
    sns.barplot(x='Svarba', y='Požymis', data=importance_df)
    plt.title('Požymių svarba')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_model_comparison(model_metrics, save_path=None):
    """
    Nubraižyti skirtingų modelių palyginimą
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    models = list(model_metrics.keys())
    
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(models):
        values = [model_metrics[model][metric] for metric in metrics]
        ax.bar(x + i * width, values, width, label=model)
    
    ax.set_ylabel('Reikšmė')
    ax.set_title('Modelių palyginimas')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(['Tikslumas', 'Preciziškumas', 'Atgaminimas', 'F1 balas'])
    ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_hyperparameter_impact(param_name, param_values, accuracies, save_path=None):
    """
    Nubraižyti hiperparametro įtaką modelio tikslumui
    """
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, accuracies, marker='o')
    plt.title(f'{param_name} įtaka modelio tikslumui')
    plt.xlabel(param_name)
    plt.ylabel('Tikslumas')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_prediction_examples(images, true_labels, pred_labels, class_names, save_path=None):
    """
    Nubraižyti prognozių pavyzdžius su tikromis ir prognozuotomis klasėmis
    """
    n_examples = min(10, len(images))
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i in range(n_examples):
        axes[i].imshow(images[i])
        axes[i].set_title(f'Tikra: {class_names[true_labels[i]]}\nPrognozė: {class_names[pred_labels[i]]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close() 