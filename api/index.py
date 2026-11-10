"""
api/index.py
Microsserviço FastAPI adaptado para Serverless na Vercel.
Detecção de Fraude em Cartão de Crédito — MegaLoja S.A.
Grupo 8 - PI V FATEC Ourinhos (Prof. Dr. Sidney Carlos Ferrari)
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="MegaLoja S.A. - Fraud Detection API (Vercel Serverless)",
    description="API de avaliação preditiva de risco de fraude em tempo real hospedada na Vercel.",
    version="1.1.0",
    docs_url="/docs",
    openapi_url="/openapi.json"
)

# Habilitar CORS para permitir chamadas de qualquer frontend/dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolução de caminho segura para ambiente Serverless
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "credit_card_fraud_model.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "model_metadata.json")

# Fallback para diretório raiz caso não esteja em api/
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "credit_card_fraud_model.joblib")
if not os.path.exists(METADATA_PATH):
    METADATA_PATH = os.path.join(os.path.dirname(BASE_DIR), "model_metadata.json")

# Carregamento do modelo serializado na inicialização
pipeline_model = None
metadata = {}

try:
    if os.path.exists(MODEL_PATH):
        pipeline_model = joblib.load(MODEL_PATH)
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
except Exception as e:
    print(f"Erro ao carregar modelo: {e}")

class TransactionInput(BaseModel):
    Time: float = Field(..., description="Segundos decorridos", example=406.0)
    Amount: float = Field(..., description="Valor monetário", example=149.62)
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
    model_name: str
    status_code: int

@app.get("/", tags=["Health Check"])
def root():
    return {
        "status": "online",
        "service": "MegaLoja Fraud Guard API (Vercel Serverless)",
        "model_loaded": pipeline_model is not None,
        "model_name": metadata.get("model_name", "MegaLoja-Fraud-Detector-Bagging-LogisticRegression"),
        "algorithm": metadata.get("algorithm", "BaggingClassifier (base_estimator=LogisticRegression)"),
        "model_version": metadata.get("version", "1.1.0"),
        "documentation": "/docs"
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_transaction(transaction: TransactionInput):
    global pipeline_model
    if pipeline_model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo preditivo não disponível no servidor."
        )

    try:
        data_dict = transaction.model_dump()
        df_input = pd.DataFrame([data_dict])
        df_input['Hour'] = (df_input['Time'] / 3600).astype(int) % 24

        expected_features = metadata.get("features", list(df_input.columns))
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
            model_name=metadata.get("model_name", "BaggingClassifier"),
            status_code=status.HTTP_200_OK
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno de inferência: {str(e)}")
