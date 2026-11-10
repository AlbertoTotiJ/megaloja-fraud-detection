"""
main.py
Microsserviço FastAPI para inferência em tempo real de fraudes em transações.
Grupo 8 - PI V - FATEC Ourinhos (Prof. Dr. Sidney Carlos Ferrari)
"""
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os
import json

app = FastAPI(
    title="MegaLoja S.A. - Fraud Detection API",
    description="API de avaliação preditiva de risco de fraude em transações de cartão de crédito.",
    version="1.0.0"
)

MODEL_PATH = "credit_card_fraud_model.joblib"
METADATA_PATH = "model_metadata.json"

pipeline_model = None
metadata = {}

@app.on_event("startup")
def load_artifacts():
    global pipeline_model, metadata
    if os.path.exists(MODEL_PATH):
        pipeline_model = joblib.load(MODEL_PATH)
        print(f"Modelo carregado de: {MODEL_PATH}")
    else:
        print(f"Aviso: {MODEL_PATH} ainda não foi gerado. Execute export_model.py!")

    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        metadata = {"version": "1.0.0", "features": ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)] + ["Hour"]}

# Esquema de Entrada
class TransactionInput(BaseModel):
    Time: float = Field(..., description="Segundos decorridos desde a primeira transação", example=406.0)
    Amount: float = Field(..., description="Valor monetário da transação", example=149.62)
    V1: float = Field(default=0.0, example=-2.31)
    V2: float = Field(default=0.0, example=1.95)
    V3: float = Field(default=0.0, example=-1.60)
    V4: float = Field(default=0.0, example=3.99)
    V5: float = Field(default=0.0, example=-0.52)
    V6: float = Field(default=0.0, example=-1.42)
    V7: float = Field(default=0.0, example=-2.53)
    V8: float = Field(default=0.0, example=1.39)
    V9: float = Field(default=0.0, example=-2.77)
    V10: float = Field(default=0.0, example=-2.77)
    V11: float = Field(default=0.0, example=3.20)
    V12: float = Field(default=0.0, example=-2.89)
    V13: float = Field(default=0.0, example=-0.59)
    V14: float = Field(default=0.0, example=-4.28)
    V15: float = Field(default=0.0, example=0.38)
    V16: float = Field(default=0.0, example=-1.14)
    V17: float = Field(default=0.0, example=-2.83)
    V18: float = Field(default=0.0, example=-0.01)
    V19: float = Field(default=0.0, example=0.41)
    V20: float = Field(default=0.0, example=0.12)
    V21: float = Field(default=0.0, example=0.51)
    V22: float = Field(default=0.0, example=-0.03)
    V23: float = Field(default=0.0, example=-0.46)
    V24: float = Field(default=0.0, example=0.32)
    V25: float = Field(default=0.0, example=0.04)
    V26: float = Field(default=0.0, example=0.17)
    V27: float = Field(default=0.0, example=0.26)
    V28: float = Field(default=0.0, example=-0.14)

class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    recommendation: str
    status_code: int

@app.get("/", tags=["Health Check"])
def root():
    return {
        "status": "online",
        "service": "MegaLoja Fraud Guard API",
        "model_loaded": pipeline_model is not None,
        "model_name": metadata.get("model_name", "MegaLoja-Fraud-Detector"),
        "algorithm": metadata.get("algorithm", "BaggingClassifier (base_estimator=LogisticRegression)"),
        "model_version": metadata.get("version", "1.1.0"),
        "documentation": "/docs"
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_transaction(transaction: TransactionInput):
    global pipeline_model
    if pipeline_model is None:
        if os.path.exists(MODEL_PATH):
            pipeline_model = joblib.load(MODEL_PATH)
        else:
            raise HTTPException(
                status_code=503,
                detail="Modelo não carregado no servidor. Execute export_model.py para gerar os artefatos."
            )
            
    try:
        # Converter payload para DataFrame
        data_dict = transaction.model_dump()
        df_input = pd.DataFrame([data_dict])
        
        # Engenharia de atributos (Hour)
        df_input['Hour'] = (df_input['Time'] / 3600).astype(int) % 24
        
        # Garantir ordenação das features
        expected_features = metadata.get("features", list(df_input.columns))
        # Reordenar se todas existirem
        if set(expected_features).issubset(set(df_input.columns)):
            df_input = df_input[expected_features]
            
        prob_fraud = float(pipeline_model.predict_proba(df_input)[0, 1])
        threshold = metadata.get("decision_threshold", 0.50)
        is_fraud = prob_fraud >= threshold
        
        if prob_fraud >= 0.75:
            recommendation = "BLOQUEIO IMEDIATO: Risco Crítico de Fraude"
        elif prob_fraud >= threshold:
            recommendation = "DESAFIO DE SEGURANÇA: Solicitar Verificação 2FA / Biometria"
        else:
            recommendation = "APROVADA: Transação com comportamento típico"
            
        return PredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=round(prob_fraud, 4),
            recommendation=recommendation,
            status_code=status.HTTP_200_OK
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno de inferência: {str(e)}")
