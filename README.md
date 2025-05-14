# Traffic Sign Recognition System

This project implements a comprehensive traffic sign recognition system using various machine learning and deep learning approaches. The system is built using Flask and includes multiple models for traffic sign classification.

## Features

- Database integration using SQLAlchemy
- Multiple model implementations (KNN, CNN, Vision Transformer)
- User interface for model training and prediction
- Support for custom image uploads
- Real-time prediction capabilities
- Comprehensive metrics and visualization
- Hyperparameter tuning interface

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set up the database:
```bash
python setup_database.py
```
4. Run the application:
```bash
python app.py
```

## Project Structure

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

## Models

1. KNN Classifier
2. CNN (Convolutional Neural Network)
3. Vision Transformer

## License

MIT License 