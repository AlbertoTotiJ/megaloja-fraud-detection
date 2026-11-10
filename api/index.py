"""
api/index.py
Sistema Completo de Detecção de Fraude — MegaLoja S.A.
FastAPI Serverless com Interface Web Integrada e API RESTful para Vercel.
Grupo 8 - PI V FATEC Ourinhos (Prof. Dr. Sidney Carlos Ferrari)
"""
import os
import json
import time
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="MegaLoja S.A. - Fraud Detection API",
    description="Sistema Preditivo de Detecção de Fraudes em Cartão de Crédito (Metodologia CRISP-DM)",
    version="1.1.0"
)

# Habilitar CORS irrestrito
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolução de caminho dos artefatos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "credit_card_fraud_model.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "model_metadata.json")

if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "credit_card_fraud_model.joblib")
if not os.path.exists(METADATA_PATH):
    METADATA_PATH = os.path.join(os.path.dirname(BASE_DIR), "model_metadata.json")

pipeline_model = None
metadata = {}

try:
    if os.path.exists(MODEL_PATH):
        pipeline_model = joblib.load(MODEL_PATH)
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
except Exception as e:
    print(f"Aviso ao carregar modelo: {e}")

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
    model_config = {"protected_namespaces": ()}
    is_fraud: bool
    fraud_probability: float
    recommendation: str
    model_name: str
    status_code: int

def run_inference(data_dict: dict):
    global pipeline_model
    if pipeline_model is None:
        raise HTTPException(status_code=503, detail="Modelo preditivo não disponível.")
    
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
        
    return {
        "is_fraud": is_fraud,
        "fraud_probability": round(prob_fraud, 4),
        "recommendation": recommendation,
        "model_name": metadata.get("model_name", "BaggingClassifier"),
        "status_code": 200
    }

HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MegaLoja S.A. — Detecção de Fraudes</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background: #0f172a; color: #f8fafc; font-family: system-ui, -apple-system, sans-serif; }
        .card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; }
        .btn-primary { background: #2563eb; border: none; }
        .btn-primary:hover { background: #1d4ed8; }
        .badge-model { background: #6366f1; padding: 6px 12px; border-radius: 8px; font-weight: 500; }
        .gauge-container { font-size: 2.5rem; font-weight: 700; }
        .form-control { background: #0f172a; border-color: #334155; color: #f8fafc; }
        .form-control:focus { background: #0f172a; border-color: #3b82f6; color: #f8fafc; box-shadow: none; }
    </style>
</head>
<body class="py-5">
    <div class="container" style="max-width: 900px;">
        <div class="text-center mb-4">
            <span class="badge badge-model mb-2">🚀 Em Produção na Vercel (CRISP-DM)</span>
            <h1 class="fw-bold">🛡️ MegaLoja S.A. — Detecção de Fraudes</h1>
            <p class="text-secondary">Projeto Integrador V — FATEC Ourinhos | Prof. Dr. Sidney Carlos Ferrari<br><strong>Grupo 8:</strong> Alberto, Geovane, Gustavo</p>
        </div>

        <div class="card p-4 mb-4 shadow">
            <h4 class="mb-3">Simulação de Transação em Tempo Real</h4>
            <div class="d-flex gap-2 mb-3">
                <button type="button" class="btn btn-outline-danger btn-sm" onclick="carregarExemploFraude()">🚨 Carregar Fraude Real</button>
                <button type="button" class="btn btn-outline-success btn-sm" onclick="carregarExemploNormal()">✅ Carregar Compra Normal</button>
            </div>

            <div class="row g-3">
                <div class="col-md-6">
                    <label class="form-label text-secondary">Tempo Decorrido (segundos):</label>
                    <input type="number" id="time_val" class="form-control" value="406.0">
                </div>
                <div class="col-md-6">
                    <label class="form-label text-secondary">Valor da Transação (R$):</label>
                    <input type="number" id="amount_val" class="form-control" value="149.62">
                </div>
            </div>

            <div class="mt-4">
                <button id="btnAvaliar" class="btn btn-primary w-100 py-3 fw-bold fs-5" onclick="avaliarTransacao()">
                    🔍 Avaliar Transação em Tempo Real
                </button>
            </div>
        </div>

        <div id="resultadoCard" class="card p-4 shadow" style="display: none;">
            <h4 class="mb-3">Resultado da Avaliação do Motor</h4>
            <div class="text-center py-3">
                <div class="text-secondary small">Probabilidade Calculada de Fraude</div>
                <div id="probDisplay" class="gauge-container text-danger">--%</div>
                <div id="decisaoDisplay" class="h5 mt-2 fw-bold">--</div>
            </div>
            <div class="bg-dark p-3 rounded small text-secondary">
                <div><strong>Modelo em Operação:</strong> <span id="modeloNome">BaggingClassifier (Regressão Logística)</span></div>
                <div><strong>Tempo de Resposta:</strong> <span id="latencia">-- ms</span></div>
            </div>
        </div>

        <div class="text-center mt-4 text-secondary small">
            <a href="/docs" class="text-info text-decoration-none me-3" target="_blank">📖 Documentação Swagger (/docs)</a> |
            <a href="https://github.com/AlbertoTotiJ/megaloja-fraud-detection" class="text-info text-decoration-none ms-3" target="_blank">🐙 Repositório GitHub</a>
        </div>
    </div>

    <script>
        const fraudValues = {
            Time: 406.0, Amount: 149.62,
            V1: -2.31, V2: 1.95, V3: -1.60, V4: 3.99, V5: -0.52, V6: -1.42, V7: -2.53, V8: 1.39,
            V9: -2.77, V10: -2.77, V11: 3.20, V12: -2.89, V13: -0.59, V14: -4.28, V15: 0.38,
            V16: -1.14, V17: -2.83, V18: -0.01, V19: 0.41, V20: 0.12, V21: 0.51, V22: -0.03,
            V23: -0.46, V24: 0.32, V25: 0.04, V26: 0.17, V27: 0.26, V28: -0.14
        };
        let currentPCA = {...fraudValues};

        function carregarExemploFraude() {
            document.getElementById('time_val').value = 406.0;
            document.getElementById('amount_val').value = 149.62;
            currentPCA = {...fraudValues};
        }

        function carregarExemploNormal() {
            document.getElementById('time_val').value = 50000.0;
            document.getElementById('amount_val').value = 45.00;
            currentPCA = { Time: 50000.0, Amount: 45.00 };
            for(let i=1; i<=28; i++) currentPCA['V' + i] = 0.0;
        }

        async function avaliarTransacao() {
            const btn = document.getElementById('btnAvaliar');
            btn.disabled = true;
            btn.innerText = "⏳ Consultando Modelo...";
            const t0 = performance.now();

            currentPCA.Time = parseFloat(document.getElementById('time_val').value) || 0;
            currentPCA.Amount = parseFloat(document.getElementById('amount_val').value) || 0;

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(currentPCA)
                });
                const data = await response.json();
                const t1 = performance.now();

                const probPercent = (data.fraud_probability * 100).toFixed(2);
                const probElem = document.getElementById('probDisplay');
                probElem.innerText = probPercent + "%";
                probElem.className = "gauge-container " + (data.is_fraud ? "text-danger" : "text-success");

                const decElem = document.getElementById('decisaoDisplay');
                decElem.innerText = (data.is_fraud ? "🚨 " : "✅ ") + data.recommendation;
                decElem.className = "h5 mt-2 fw-bold " + (data.is_fraud ? "text-danger" : "text-success");

                document.getElementById('latencia').innerText = Math.round(t1 - t0) + " ms";
                document.getElementById('modeloNome').innerText = data.model_name;
                document.getElementById('resultadoCard').style.display = 'block';
            } catch (err) {
                alert("Erro ao consultar API: " + err);
            } finally {
                btn.disabled = false;
                btn.innerText = "🔍 Avaliar Transação em Tempo Real";
            }
        }
    </script>
</body>
</html>
"""

SWAGGER_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>MegaLoja S.A. - Swagger UI</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script>
        SwaggerUIBundle({
            url: '/openapi.json',
            dom_id: '#swagger-ui',
            presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
            layout: "BaseLayout"
        });
    </script>
</body>
</html>
"""

# Resposta JSON do status da API
def get_health_json():
    return {
        "status": "online",
        "service": "MegaLoja Fraud Guard API (Vercel Serverless)",
        "model_loaded": pipeline_model is not None,
        "model_name": metadata.get("model_name", "MegaLoja-Fraud-Detector-Bagging-LogisticRegression"),
        "algorithm": metadata.get("algorithm", "BaggingClassifier (base_estimator=LogisticRegression)"),
        "model_version": metadata.get("version", "1.1.0"),
        "documentation": "/docs"
    }

# 1. Rota raiz: serve o Dashboard HTML interativo se for navegador, ou JSON se for API
@app.get("/", response_class=HTMLResponse)
@app.get("/index", response_class=HTMLResponse)
@app.get("/api/index.py", response_class=HTMLResponse)
def root_view(request: Request):
    accept = request.headers.get("accept", "")
    if "text/html" in accept or "*/*" in accept:
        return HTMLResponse(content=HTML_DASHBOARD)
    return JSONResponse(content=get_health_json())

# 2. Endpoint exclusivo de Health Check em JSON
@app.get("/health")
@app.get("/api/health")
def health_endpoint():
    return get_health_json()

# 3. Documentação Swagger
@app.get("/docs", response_class=HTMLResponse)
@app.get("/api/docs", response_class=HTMLResponse)
@app.get("/api/index.py/docs", response_class=HTMLResponse)
def docs_view():
    return HTMLResponse(content=SWAGGER_HTML)

@app.get("/openapi.json")
@app.get("/api/openapi.json")
@app.get("/api/index.py/openapi.json")
def openapi_view():
    return app.openapi()

# 4. Endpoint de Inferência Preditiva (POST)
@app.post("/predict", response_model=PredictionResponse)
@app.post("/api/predict", response_model=PredictionResponse)
@app.post("/api/index.py/predict", response_model=PredictionResponse)
def predict_endpoint(transaction: TransactionInput):
    res = run_inference(transaction.model_dump())
    return PredictionResponse(**res)

# 5. Roteador Universal Catch-All
@app.api_route("/{full_path:path}", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def universal_router(request: Request, full_path: str):
    clean_path = full_path.strip("/")
    
    # Se for predição
    if request.method == "POST":
        try:
            body = await request.json()
            return PredictionResponse(**run_inference(body))
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
            
    # Se for documentação
    if "doc" in clean_path:
        return HTMLResponse(content=SWAGGER_HTML)
    if "openapi" in clean_path:
        return app.openapi()
    if "health" in clean_path or "json" in clean_path:
        return JSONResponse(content=get_health_json())
        
    # Navegador abrindo qualquer link
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return HTMLResponse(content=HTML_DASHBOARD)
        
    return JSONResponse(content=get_health_json())
