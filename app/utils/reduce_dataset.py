import os
import random
import shutil
from pathlib import Path

def sumažinti_duomenų_kiekį(base_dir='duomenys/Train', target_count=50):
    """
    Sumažina kiekvienos klasės paveikslėlių kiekį iki nurodyto skaičiaus.
    
    Args:
        base_dir (str): Pagrindinis katalogas su klasėmis
        target_count (int): Kiek paveikslėlių palikti kiekvienoje klasėje
    """
    base_path = Path(base_dir)
    
    # Sukurkime atsarginę kopiją
    backup_dir = base_path.parent / 'Train_backup'
    if not backup_dir.exists():
        print(f"Kuriama atsarginė kopija į {backup_dir}...")
        shutil.copytree(base_path, backup_dir)
        print("Atsarginė kopija sukurta!")
    
    # Apdorokime kiekvieną klasę
    for class_dir in base_path.iterdir():
        if not class_dir.is_dir():
            continue
            
        # Gauti visus paveikslėlius
        images = list(class_dir.glob('*.png'))  # arba *.jpg, priklausomai nuo formato
        current_count = len(images)
        
        if current_count > target_count:
            print(f"Klasė {class_dir.name}: {current_count} -> {target_count} paveikslėlių")
            
            # Atsitiktinai išrinkti paveikslėlius, kuriuos palikti
            keep_images = random.sample(images, target_count)
            
            # Ištrinti likusius paveikslėlius
            for img in images:
                if img not in keep_images:
                    img.unlink()
        else:
            print(f"Klasė {class_dir.name}: {current_count} paveikslėlių (nereikia mažinti)")

if __name__ == '__main__':
    sumažinti_duomenų_kiekį() 