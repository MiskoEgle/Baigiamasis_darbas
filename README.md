# Kelių ženklų atpažinimo sistema

Šis projektas įgyvendina kelių ženklų atpažinimo sistemą, naudojančią įvairius mašininio mokymosi ir giluminio mokymosi modelius. Sistema sukurta su Flask ir apima kelis modelius ženklų klasifikavimui.

## Savybės

- Duomenų bazės integracija su SQLAlchemy
- Keli modeliai (KNN, CNN, Vision Transformer)
- Vartotojo sąsaja modelio mokymui ir prognozavimui
- Palaikomas individualių paveikslėlių įkėlimas
- Realus laiko prognozavimas
- Išsamios metrikos ir vizualizacijos
- Hiperparametrų derinimo sąsaja
- **Automatinis paveikslėlių apdorojimas** (dydis, spalvos, tipas) backend'e
- **Lietuviška vartotojo sąsaja ir grafikai**

## Diegimas

1. Nukopijuokite (clone) šį repozitoriją
2. Įdiekite priklausomybes:
```bash
pip install -r requirements.txt
```
3. Paruoškite duomenų bazę:
```bash
python setup_database.py
```
4. Paleiskite aplikaciją:
```bash
python app.py
```

## Naudojimas

### Modelio mokymas
- Eikite į puslapį **Modelio mokymas**.
- Įkelkite CSV arba ZIP failą su mokymo duomenimis.
- Pasirinkite modelio tipą (KNN, CNN, Transformer).
- Paspauskite **Mokyti**. Modelis bus apmokytas, o rezultatai ir grafikai bus sugeneruoti automatiškai.

### Prognozavimas
- Eikite į pagrindinį puslapį arba **Pradžia**.
- Įkelkite norimą paveikslėlį (gali būti bet kokio formato: spalvotas, pilkas, skirtingo dydžio).
- Pasirinkite modelio tipą.
- Paspauskite **Prognozuoti**. Sistema automatiškai apdoros paveikslėlį (konvertuos į 32x32, RGB, uint8, normalizuos jei reikia) ir pateiks prognozę.

### Grafikai ir rezultatai
- Visi mokymo ir validacijos grafikai, sumaišties matricos, klasių pasiskirstymai ir kt. automatiškai sugeneruojami ir pasiekiami per meniu **Grafikai**.
- Rezultatus galima peržiūrėti ir JSON formatu.

### Kiti svarbūs pakeitimai
- **Automatinis paveikslėlių apdorojimas**: nereikia rūpintis formatu ar tipu – sistema viską padaro už jus.
- **Klaidos dėl paveikslėlio tipo** (OpenCV HOG/Haar): išspręstos automatiškai backend'e.
- **Lietuviški komentarai ir grafikai**: visa sąsaja ir vizualizacijos lietuvių kalba.

## Projekto struktūra

```
├── app/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database.py
│   │   ├── knn_model.py
│   │   ├── cnn_model.py
│   │   └── transformer_model.py
│   ├── static/
│   │   ├── css/
│   │   └── js/
│   ├── templates/
│   │   ├── base.html
│   │   ├── index.html
│   │   └── results.html
│   └── utils/
│       ├── __init__.py
│       ├── data_processing.py
│       └── visualization.py
├── data/
│   ├── raw/
│   └── processed/
├── tests/
├── app.py
├── config.py
├── requirements.txt
└── README.md
```

## Modeliai

1. KNN klasifikatorius
2. CNN (konvoliucinis neuroninis tinklas)
3. Vision Transformer

## Licencija

MIT License 