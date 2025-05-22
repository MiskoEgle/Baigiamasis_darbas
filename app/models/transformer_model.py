import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from app.utils.visualization import (
    sukurti_konfuzijos_macica,
    sukurti_klasiu_grafikus,
    sukurti_metriku_grafikus,
    issaugoti_rezultatus
)
from app.model_progress import model_progress

class TransformerNet(nn.Module):
    """
    Transformer modelio klasė, skirta paveikslėlių klasifikacijai.
    Naudoja transformer architektūrą su poziciniu koduojimu ir globaliu vidurkiu.
    """
    def __init__(self, input_dim, num_classes, d_model=256, nhead=8, num_layers=6):
        """
        Inicializuoja Transformer modelį.
        
        Args:
            input_dim: Įvesties dimensija (paveikslėlio požymių skaičius)
            num_classes: Klasifikacijos klasių skaičius
            d_model: Modelio dimensija (256 pagal nutylėjimą)
            nhead: Dėmesio galvų skaičius (8 pagal nutylėjimą)
            num_layers: Transformer sluoksnių skaičius (6 pagal nutylėjimą)
        """
        super().__init__()
        self.logger = logging.getLogger('app')
        
        # Sukuriame modelio komponentus
        self.embedding = nn.Linear(input_dim, d_model)  # Požymių įterpimas į d_model dimensiją
        self.pos_encoder = PositionalEncoding(d_model)  # Pozicinis koduojimas
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)  # Transformer koduotuvo sluoksniai
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)  # Transformer koduotuvas
        self.decoder = nn.Linear(d_model, num_classes)  # Išvesties sluoksnis
        
    def forward(self, x):
        """
        Pirmyn sklidimo funkcija.
        
        Args:
            x: Įvesties duomenys [batch_size, features]
            
        Returns:
            Modelio išvestis [batch_size, num_classes]
        """
        # Pridedame sekvencijos dimensiją, jei jos nėra
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.embedding(x)  # Požymių įterpimas
        x = self.pos_encoder(x)  # Pozicinis koduojimas
        x = self.transformer_encoder(x)  # Transformer apdorojimas
        x = x.mean(dim=1)  # Globalus vidurkis
        x = self.decoder(x)  # Klasifikacijos sluoksnis
        return x
        
class PositionalEncoding(nn.Module):
    """
    Pozicinis koduojimo klasė, kuri prideda pozicijos informaciją prie įvesties.
    Naudoja sinusoidinį koduojimą.
    """
    def __init__(self, d_model, max_len=5000):
        """
        Inicializuoja pozicinį koduojimą.
        
        Args:
            d_model: Modelio dimensija
            max_len: Maksimali sekvencijos ilgis
        """
        super().__init__()
        # Sukuriame pozicinio koduojimo matricą
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Sinusoidinis koduojimas lyginėms pozicijoms
        pe[:, 1::2] = torch.cos(position * div_term)  # Kosinusoidinis koduojimas nelyginėms pozicijoms
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Prideda pozicinį koduojimą prie įvesties.
        
        Args:
            x: Įvesties duomenys
            
        Returns:
            Duomenys su pridėtu poziciniu koduojimu
        """
        x = x + self.pe[:, :x.size(1)]
        return x
        
class TransformerModelis:
    """
    Transformer modelio apvalkalas, kuris tvarko modelio mokymą, prognozavimą ir išsaugojimą.
    """
    def __init__(self):
        """
        Inicializuoja Transformer modelio apvalkalą.
        Nustato įrenginį (GPU arba CPU) ir modelio parametrus.
        """
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger('app')
        self.input_dim = 3072  # 32*32*3 RGB paveikslėlio požymiai
        self.num_classes = 2  # Klasifikacijos klasių skaičius
        
    def mokyti(self, X, y):
        """
        Mokyti Transformer modelį.
        
        Args:
            X: Mokymo duomenys
            y: Mokymo etiketės
            
        Returns:
            dict: Mokymo rezultatai ir metrikos
        """
        try:
            # Padalinti duomenis į mokymo ir testavimo aibes
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )
            n_epochs = 50  # Mokymo epochų skaičius
            model_progress.start('transformer', n_epochs)
            
            # Konvertuoti duomenis į PyTorch tensorius
            X_train = torch.FloatTensor(X_train)
            y_train = torch.LongTensor(y_train)
            X_test = torch.FloatTensor(X_test)
            y_test = torch.LongTensor(y_test)
            
            # Sukurti duomenų įkrovėjus
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            # Sukurti ir perkelti modelį į atitinkamą įrenginį
            self.model = TransformerNet(
                input_dim=self.input_dim,
                num_classes=self.num_classes
            ).to(self.device)
            
            # Mokymo parametrai
            criterion = nn.CrossEntropyLoss()  # Nuostolių funkcija
            optimizer = torch.optim.Adam(self.model.parameters())  # Optimizatorius
            
            # Mokymo ciklas
            self.model.train()
            for epoch in range(n_epochs):
                if model_progress.should_stop('transformer'):
                    model_progress.error('transformer', 'Mokymas sustabdytas vartotojo')
                    return None
                    
                for batch_X, batch_y in train_loader:
                    if model_progress.should_stop('transformer'):
                        model_progress.error('transformer', 'Mokymas sustabdytas vartotojo')
                        return None
                        
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    optimizer.zero_grad()  # Išvalyti gradientus
                    outputs = self.model(batch_X)  # Pirmyn sklidimas
                    loss = criterion(outputs, batch_y)  # Apskaičiuoti nuostolį
                    loss.backward()  # Atgal sklidimas
                    optimizer.step()  # Atnaujinti svorius
                    
                model_progress.update('transformer', epoch+1, message=f"Transformer epocha {epoch+1}/{n_epochs}")
            model_progress.finish('transformer')

            # Išsaugoti modelį
            self.issaugoti_modeli('app/static/models/transformer_model')

            # Gauti prognozes testavimo aibei
            self.model.eval()
            with torch.no_grad():
                X_test = X_test.to(self.device)
                outputs = self.model(X_test)
                y_pred = outputs.argmax(dim=1).cpu().numpy()
            
            # Apskaičiuoti metrikas
            accuracy = accuracy_score(y_test, y_pred)  # Tikslumas
            f1 = f1_score(y_test, y_pred, average='weighted')  # F1 balas
            precision = precision_score(y_test, y_pred, average='weighted')  # Preciziškumas
            recall = recall_score(y_test, y_pred, average='weighted')  # Atgaminimas
            
            # Apskaičiuoti kiekvienos klasės tikslumą
            klasiu_tikslumas = []
            for klase in np.unique(y):
                mask = y_test == klase
                klasiu_tikslumas.append(accuracy_score(y_test[mask], y_pred[mask]))
            
            # Sukurti vizualizacijas
            sukurti_konfuzijos_macica(
                y_test,
                y_pred,
                np.unique(y),
                'Transformer',
                'app/static/transformer_grafikai/konfuzijos_macica.png'
            )
            
            sukurti_klasiu_grafikus(
                np.unique(y),
                klasiu_tikslumas,
                'Transformer',
                'app/static/transformer_grafikai/klasiu_grafikai.png'
            )
            
            sukurti_metriku_grafikus(
                {
                    'Tikslumas': float(accuracy),
                    'F1 balas': float(f1),
                    'Precision': float(precision),
                    'Recall': float(recall)
                },
                'Transformer',
                'app/static/transformer_grafikai/metriku_grafikai.png'
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
                'app/static/transformer_porciju_rezultatai.json'
            )
            
            self.logger.info("Transformer modelis sėkmingai išmokytas")
            return results
            
        except Exception as e:
            self.logger.error(f"Klaida mokant Transformer modelį: {str(e)}")
            raise
            
    def prognozuoti(self, X):
        """
        Prognozuoti klasę ir grąžinti prognozę bei tikimybę.
        
        Args:
            X: Įvesties duomenys
            
        Returns:
            tuple: (prognozuota klasė, tikimybė)
        """
        try:
            if self.model is None:
                raise ValueError("Modelis nėra išmokytas")
                
            # Paruošti įvesties duomenis
            X = np.array(X)
            if X.ndim == 3:
                X = X.flatten()  # Išlyginti 3D masyvą
            if X.ndim == 1:
                X = X[None, :]  # Pridėti batch dimensiją
                
            X = torch.FloatTensor(X).to(self.device)
            
            # Gauti prognozę
            self.model.eval()
            with torch.no_grad():
                output = self.model(X)
                tikimybes = torch.softmax(output, dim=1).cpu().numpy()[0]
                prognoze = int(np.argmax(tikimybes))
                tikimybe = float(np.max(tikimybes))
                return prognoze, tikimybe
                
        except Exception as e:
            self.logger.error(f"Klaida prognozuojant su Transformer modeliu: {str(e)}")
            raise
            
    def issaugoti_modeli(self, kelias):
        """
        Išsaugoti modelį į nurodytą kelią.
        
        Args:
            kelias: Kelias, kur išsaugoti modelį
        """
        try:
            if self.model is None:
                raise ValueError("Modelis nėra išmokytas")
                
            os.makedirs(os.path.dirname(kelias), exist_ok=True)
            torch.save(self.model.state_dict(), kelias + '.pt')
            
            self.logger.info(f"Transformer modelis sėkmingai išsaugotas: {kelias}")
            
        except Exception as e:
            self.logger.error(f"Klaida išsaugant Transformer modelį: {str(e)}")
            raise
            
    def uzsikrauti_modeli(self, kelias):
        """
        Užsikrauti modelį iš nurodyto kelio.
        
        Args:
            kelias: Kelias, iš kur užsikrauti modelį
        """
        try:
            # Sukurti naują modelį su tais pačiais parametrais
            self.input_dim = 3072  # 32*32*3 RGB paveikslėlio požymiai
            self.num_classes = 2  # Klasifikacijos klasių skaičius
            self.model = TransformerNet(
                input_dim=self.input_dim,
                num_classes=self.num_classes
            ).to(self.device)
            
            # Užsikrauti išsaugotus svorius
            if not kelias.endswith('.pt'):
                kelias += '.pt'
            self.model.load_state_dict(torch.load(kelias, map_location=self.device))
            self.model.eval()
            
            self.logger.info(f"Transformer modelis sėkmingai užsikrautas: {kelias}")
            
        except Exception as e:
            self.logger.error(f"Klaida užsikraunant Transformer modelį: {str(e)}")
            raise 