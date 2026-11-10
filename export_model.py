"""
export_model.py
Treina e serializa o pipeline completo de detecção de fraude para produção.
Grupo 8 - PI V - FATEC Ourinhos (Prof. Dr. Sidney Carlos Ferrari)
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

def load_or_create_sample_data(csv_path="creditcard.csv"):
    if os.path.exists(csv_path):
        print(f"Carregando dataset real a partir de: {csv_path}")
        df = pd.read_csv(csv_path).dropna()
    else:
        print(f"Aviso: Arquivo '{csv_path}' não encontrado no diretório local.")
        print("Gerando amostra sintética estruturada idêntica ao dataset da Kaggle para viabilizar teste de deploy...")
        np.random.seed(42)
        n_samples = 5000
        n_frauds = 50
        
        # 5000 transações sintéticas (4950 legítimas, 50 fraudes)
        data = {
            'Time': np.random.uniform(0, 172800, n_samples),
            'Amount': np.random.exponential(88.0, n_samples),
            'Class': np.array([0] * (n_samples - n_frauds) + [1] * n_frauds)
        }
        for i in range(1, 29):
            data[f'V{i}'] = np.random.normal(0, 1, n_samples)
            
        df = pd.DataFrame(data).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Amostra criada com {len(df)} transações ({df['Class'].sum()} fraudes).")
    
    # Engenharia de atributos idêntica ao notebook
    df['Hour'] = (df['Time'] / 3600).astype(int) % 24
    return df

def train_and_export():
    df = load_or_create_sample_data()
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Holdout de validação estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print("Construindo Pipeline de Produção (RobustScaler + RandomUnderSampler + BaggingClassifier(LogisticRegression))...")
    pipeline = Pipeline(steps=[
        ('scaler', RobustScaler()),
        ('sampler', RandomUnderSampler(random_state=42)),
        ('classifier', BaggingClassifier(
            estimator=LogisticRegression(max_iter=1000, random_state=42),
            n_estimators=10,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    print("Ajustando modelo com dados de treino...")
    pipeline.fit(X_train, y_train)
    
    score_acc = pipeline.score(X_test, y_test)
    print(f"Modelo treinado com sucesso! Score de teste: {score_acc:.4f}")
    
    # Serialização do Artefato
    model_filename = 'credit_card_fraud_model.joblib'
    joblib.dump(pipeline, model_filename, compress=3)
    print(f"Artefato serializado salvo como: {model_filename}")
    
    # Metadados do Modelo para Governança
    metadata = {
        "model_name": "MegaLoja-Fraud-Detector-Bagging-LogisticRegression",
        "algorithm": "BaggingClassifier (base_estimator=LogisticRegression)",
        "version": "1.1.0",
        "creation_date": datetime.now().isoformat(),
        "features": list(X.columns),
        "target": "Class (0: Legítima, 1: Fraude)",
        "author": "Grupo 8 - PI V FATEC Ourinhos",
        "decision_threshold": 0.50
    }
    with open('model_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    print("Metadados exportados com sucesso em model_metadata.json")

if __name__ == '__main__':
    train_and_export()
