"""
api/index.py
Sistema Completo de Detecção de Fraude — MegaLoja S.A.
FastAPI Serverless com Interface Web Integrada (Design System Dark OLED), API RESTful e Monitoramento de Drift (Etapa 5 CRISP-DM).
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

# Importação do módulo de drift
try:
    from drift_monitor import run_drift_analysis
except ImportError:
    try:
        from api.drift_monitor import run_drift_analysis
    except Exception:
        run_drift_analysis = None

app = FastAPI(
    title="MegaLoja S.A. - Fraud Detection API",
    description="Sistema Preditivo de Detecção de Fraudes em Cartão de Crédito (Metodologia CRISP-DM)",
    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    V27: float = Field(default=0.26, example=0.26)
    V28: float = Field(default=-0.14, example=-0.14)

class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    is_fraud: bool
    fraud_probability: float
    recommendation: str
    risk_zone: str = "success"
    action_type: str = "approved"
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
        recommendation = "Transação Negada por Política de Segurança"
        risk_zone = "danger"
        action_type = "blocked"
    elif prob_fraud >= threshold:
        recommendation = "Atenção: Insira o código SMS enviado ao seu celular"
        risk_zone = "warning"
        action_type = "challenge_sms_2fa"
    else:
        recommendation = "Pagamento Aprovado"
        risk_zone = "success"
        action_type = "approved"
        
    return {
        "is_fraud": is_fraud,
        "fraud_probability": round(prob_fraud, 4),
        "recommendation": recommendation,
        "risk_zone": risk_zone,
        "action_type": action_type,
        "model_name": metadata.get("model_name", "BaggingClassifier"),
        "status_code": 200
    }

HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MegaLoja Tech Store — Vitrine E-Commerce & Prevenção Antifraude</title>
    <!-- Google Fonts: Fira Code + Fira Sans -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        :root {
            --color-primary: #0F172A;
            --color-on-primary: #FFFFFF;
            --color-secondary: #1E293B;
            --color-on-secondary: #FFFFFF;
            --color-accent: #22C55E;
            --color-on-accent: #0F172A;
            --color-background: #020617;
            --color-foreground: #F8FAFC;
            --color-card: #0E1223;
            --color-card-foreground: #F8FAFC;
            --color-muted: #1A1E2F;
            --color-muted-foreground: #94A3B8;
            --color-border: #334155;
            --color-destructive: #EF4444;
            --color-on-destructive: #FFFFFF;
            --color-warning: #F59E0B;
            --color-info: #38BDF8;
            --shadow-sm: 0 1px 2px rgba(0,0,0,0.4);
            --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.5), 0 2px 4px -2px rgba(0,0,0,0.4);
            --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.6), 0 4px 6px -4px rgba(0,0,0,0.5);
            --shadow-glow-accent: 0 0 18px rgba(34, 197, 94, 0.3);
            --shadow-glow-danger: 0 0 18px rgba(239, 68, 68, 0.4);
            --shadow-glow-warning: 0 0 18px rgba(245, 158, 11, 0.35);
        }

        * { box-sizing: border-box; }
        body {
            background-color: var(--color-background);
            color: var(--color-foreground);
            font-family: 'Fira Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            min-height: 100vh;
            margin: 0;
            padding: 0;
            -webkit-font-smoothing: antialiased;
        }

        h1, h2, h3, h4, h5, h6, .font-mono, .metric-val, .badge-mono {
            font-family: 'Fira Code', monospace;
            letter-spacing: -0.02em;
        }

        /* Ambient Glow Header Background */
        .header-glow {
            background: radial-gradient(circle at 50% -20%, rgba(34, 197, 94, 0.12) 0%, rgba(2, 6, 23, 0) 70%);
            border-bottom: 1px solid var(--color-border);
            padding: 1.8rem 1rem 1.4rem;
        }

        .card-custom {
            background: var(--color-card);
            border: 1px solid var(--color-border);
            border-radius: 12px;
            box-shadow: var(--shadow-md);
            transition: border-color 200ms ease, box-shadow 200ms ease, transform 200ms ease;
        }
        .card-custom:hover {
            border-color: #475569;
        }

        /* Product Cards */
        .product-card {
            background: var(--color-card);
            border: 1px solid var(--color-border);
            border-radius: 12px;
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            height: 100%;
            transition: all 200ms ease;
        }
        .product-card:hover {
            border-color: var(--color-accent);
            box-shadow: var(--shadow-glow-accent);
            transform: translateY(-3px);
        }
        .product-img-box {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
            border: 1px solid var(--color-border);
            border-radius: 10px;
            height: 110px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 0.85rem;
            position: relative;
        }

        /* Virtual Credit Card */
        .virtual-card {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #070b14 100%);
            border: 1px solid rgba(34, 197, 94, 0.35);
            border-radius: 14px;
            padding: 1.25rem;
            color: #fff;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
            position: relative;
            overflow: hidden;
        }
        .virtual-card::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(34, 197, 94, 0.08) 0%, transparent 60%);
            pointer-events: none;
        }
        .chip-sim {
            width: 40px;
            height: 28px;
            background: linear-gradient(135deg, #fbbf24 0%, #d97706 100%);
            border-radius: 6px;
            box-shadow: inset 0 1px 2px rgba(255,255,255,0.4);
        }

        /* Preset Cards */
        .preset-chip {
            background: var(--color-muted);
            border: 1px solid var(--color-border);
            border-radius: 8px;
            padding: 10px 14px;
            cursor: pointer;
            transition: all 200ms ease;
            text-align: left;
            width: 100%;
        }
        .preset-chip:hover {
            border-color: var(--color-info);
            background: #22293f;
        }
        .preset-chip.active {
            border-color: var(--color-accent);
            background: rgba(34, 197, 94, 0.1);
        }

        /* Form Inputs */
        .form-label {
            color: var(--color-muted-foreground);
            font-size: 0.85rem;
            font-weight: 500;
            margin-bottom: 4px;
        }
        .form-control-custom {
            background: var(--color-primary);
            border: 1px solid var(--color-border);
            border-radius: 8px;
            color: var(--color-foreground);
            font-family: 'Fira Code', monospace;
            padding: 10px 14px;
            width: 100%;
            transition: border-color 200ms ease, box-shadow 200ms ease;
        }
        .form-control-custom:focus {
            outline: none;
            border-color: var(--color-accent);
            box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.2);
            background: var(--color-primary);
            color: #fff;
        }

        /* Buttons */
        .btn-cta {
            background: var(--color-accent);
            color: #04130a;
            font-weight: 700;
            border: none;
            border-radius: 8px;
            padding: 12px 20px;
            cursor: pointer;
            transition: all 200ms ease;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            font-family: 'Fira Sans', sans-serif;
            font-size: 1rem;
        }
        .btn-cta:hover:not(:disabled) {
            background: #16a34a;
            box-shadow: var(--shadow-glow-accent);
            transform: translateY(-1px);
        }
        .btn-cta:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .btn-secondary-custom {
            background: var(--color-secondary);
            color: var(--color-foreground);
            border: 1px solid var(--color-border);
            border-radius: 8px;
            padding: 9px 15px;
            cursor: pointer;
            transition: all 200ms ease;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        .btn-secondary-custom:hover {
            background: #283548;
            border-color: var(--color-info);
            color: #fff;
        }

        /* Badges */
        .badge-status {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            font-family: 'Fira Code', monospace;
        }
        .badge-success { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }
        .badge-danger { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
        .badge-warning { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
        .badge-info { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }
        .badge-neutral { background: var(--color-muted); color: var(--color-muted-foreground); border: 1px solid var(--color-border); }

        .badge-pill-custom {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            font-family: 'Fira Code', monospace;
        }
        .badge-green { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }
        .badge-yellow { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
        .badge-red { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
        .badge-blue { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }
        .form-select-custom {
            background: var(--color-primary);
            border: 1px solid var(--color-border);
            border-radius: 8px;
            color: var(--color-foreground);
            font-family: 'Fira Code', monospace;
            padding: 8px 12px;
            width: 100%;
        }
        .form-select-custom:focus {
            outline: none;
            border-color: var(--color-accent);
            box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.25);
        }

        /* XAI Explainable AI Styles */
        .xai-card {
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid var(--color-border);
            border-radius: 12px;
            box-shadow: var(--shadow-glow-muted);
        }
        .xai-bar-track {
            height: 8px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 4px;
            overflow: hidden;
            width: 100%;
        }
        .xai-bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 400ms ease;
        }
        .xai-bar-danger { background: var(--color-destructive); }
        .xai-bar-success { background: var(--color-accent); }
        .xai-bar-warning { background: var(--color-warning); }

        /* Pulse dot */
        .pulse-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }
        .pulse-dot.online {
            background-color: #22c55e;
            box-shadow: 0 0 8px #22c55e;
            animation: pulseAnim 2s infinite;
        }
        @keyframes pulseAnim {
            0% { transform: scale(0.95); opacity: 0.8; }
            50% { transform: scale(1.2); opacity: 1; }
            100% { transform: scale(0.95); opacity: 0.8; }
        }

        /* Tabs */
        .nav-tabs-custom {
            display: flex;
            gap: 8px;
            border-bottom: 1px solid var(--color-border);
            padding-bottom: 0;
            margin-bottom: 24px;
        }
        .tab-btn {
            background: transparent;
            border: none;
            border-bottom: 3px solid transparent;
            color: var(--color-muted-foreground);
            padding: 12px 18px;
            font-weight: 600;
            cursor: pointer;
            transition: all 200ms ease;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 0.95rem;
        }
        .tab-btn:hover {
            color: var(--color-foreground);
            background: rgba(30, 41, 59, 0.4);
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
        }
        .tab-btn.active {
            color: var(--color-accent);
            border-bottom-color: var(--color-accent);
            background: rgba(34, 197, 94, 0.05);
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
        }

        /* SVG Icons */
        .icon {
            width: 18px;
            height: 18px;
            fill: none;
            stroke: currentColor;
            stroke-width: 2;
            stroke-linecap: round;
            stroke-linejoin: round;
            flex-shrink: 0;
        }
        .icon-lg {
            width: 24px;
            height: 24px;
        }
        .icon-xl {
            width: 48px;
            height: 48px;
        }

        /* Tables */
        .table-custom {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
            color: var(--color-foreground);
        }
        .table-custom th {
            background: var(--color-primary);
            color: var(--color-muted-foreground);
            font-family: 'Fira Code', monospace;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
            padding: 10px 12px;
            border-bottom: 1px solid var(--color-border);
            text-align: left;
        }
        .table-custom td {
            padding: 10px 12px;
            border-bottom: 1px solid #1e293b;
            font-family: 'Fira Code', monospace;
        }
        .table-custom tr:hover td {
            background: rgba(30, 41, 59, 0.5);
        }

        /* Gauge progress bar */
        .gauge-bar-wrapper {
            background: var(--color-primary);
            height: 12px;
            border-radius: 6px;
            overflow: hidden;
            border: 1px solid var(--color-border);
            margin: 10px 0;
        }
        .gauge-bar-fill {
            height: 100%;
            transition: width 500ms cubic-bezier(0.16, 1, 0.3, 1), background-color 300ms ease;
        }

        /* Accessibility: keyboard focus */
        a:focus-visible, button:focus-visible, input:focus-visible {
            outline: 2px solid var(--color-accent);
            outline-offset: 2px;
        }
    </style>
</head>
<body>

    <!-- Header -->
    <header class="header-glow">
        <div class="container" style="max-width: 1050px;">
            <div class="d-flex flex-wrap justify-content-between align-items-center gap-3">
                <div class="d-flex align-items-center gap-3">
                    <div style="background: rgba(34, 197, 94, 0.15); border: 1px solid rgba(34, 197, 94, 0.3); border-radius: 10px; padding: 10px; color: var(--color-accent);">
                        <svg class="icon-lg" viewBox="0 0 24 24"><path d="M6 2 3 6v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6l-3-4Z"/><path d="M3 6h18"/><path d="M16 10a4 4 0 0 1-8 0"/></svg>
                    </div>
                    <div>
                        <div class="d-flex align-items-center gap-2">
                            <h1 class="h4 mb-0 fw-bold text-white">MegaLoja Tech Store</h1>
                            <span class="badge-status badge-success">
                                <span class="pulse-dot online"></span> AntiFraude Ativo
                            </span>
                        </div>
                        <p class="text-secondary small mb-0">Vitrine E-Commerce &bull; Detecção de Fraude em Tempo Real (Metodologia CRISP-DM)</p>
                    </div>
                </div>
                <div class="text-end">
                    <div class="d-flex align-items-center gap-2 mb-1 justify-content-end">
                        <span class="badge-status badge-neutral">Vercel Serverless</span>
                        <span class="badge-status badge-neutral">Bagging (RL x10)</span>
                    </div>
                    <span class="text-secondary small font-mono">FATEC Ourinhos • Prof. Dr. Sidney C. Ferrari</span>
                </div>
            </div>
        </div>
    </header>

    <!-- Main Container -->
    <main class="container py-4" style="max-width: 1050px;">

        <!-- Navigation Tabs (3 Passos: 1. Selecionar Produto, 2. Checkout, 3. Conclusão) -->
        <nav class="nav-tabs-custom" role="tablist" aria-label="Etapas da Loja">
            <button class="tab-btn active" id="tab-btn-produtos" onclick="switchTab('produtos')" role="tab">
                <svg class="icon" viewBox="0 0 24 24"><rect width="20" height="14" x="2" y="3" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
                1. Selecionar Produto
            </button>
            <button class="tab-btn" id="tab-btn-checkout" onclick="switchTab('checkout')" role="tab">
                <svg class="icon" viewBox="0 0 24 24"><rect width="20" height="14" x="2" y="5" rx="2"/><line x1="2" y1="10" x2="22" y2="10"/></svg>
                2. Checkout
            </button>
            <button class="tab-btn" id="tab-btn-conclusao" onclick="switchTab('conclusao')" role="tab">
                <svg class="icon" viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/></svg>
                3. Conclusão
            </button>
        </nav>

        <!-- ========================================== -->
        <!-- TAB 1: 1. SELECIONAR PRODUTO -->
        <!-- ========================================== -->
        <section id="tab-produtos" role="tabpanel">
            <div class="card-custom p-3 mb-4 d-flex flex-wrap justify-content-between align-items-center gap-2" style="background: rgba(15, 23, 42, 0.7);">
                <div>
                    <span class="badge-status badge-info mb-1 font-mono">PASSO 1 DE 3 • CATÁLOGO DA LOJA</span>
                    <h2 class="h5 mb-0 text-white fw-bold">Selecione um produto para iniciar a simulação de compra</h2>
                    <p class="text-secondary small mb-0">Cada produto possui valor e características enviadas ao pipeline preditivo no Checkout.</p>
                </div>
                <span class="badge-status badge-success">8 Itens Disponíveis</span>
            </div>

            <!-- Grid de Produtos da Loja -->
            <div class="row g-3" id="catalogoProdutosContainer"></div>
        </section>

        <!-- ========================================== -->
        <!-- TAB 2: 2. CHECKOUT -->
        <!-- ========================================== -->
        <section id="tab-checkout" role="tabpanel" style="display: none;">
            <div class="card-custom p-3 mb-4 d-flex flex-wrap justify-content-between align-items-center gap-2" style="background: rgba(15, 23, 42, 0.7);">
                <div>
                    <span class="badge-status badge-info mb-1 font-mono">PASSO 2 DE 3 • CHECKOUT SEGURO</span>
                    <h2 class="h5 mb-0 text-white fw-bold">Revise seu pedido e configure o cenário de risco antifraude</h2>
                    <p class="text-secondary small mb-0">O modelo BaggingClassifier processará a requisição em tempo real via API FastAPI.</p>
                </div>
                <button type="button" class="btn-secondary-custom btn-sm" onclick="switchTab('produtos')">
                    &larr; Trocar de Produto
                </button>
            </div>

            <div class="row g-4">
                <!-- Coluna 1: Resumo do Pedido & Cartão Simulado -->
                <div class="col-lg-6">
                    <div class="card-custom p-4 mb-4">
                        <div class="d-flex justify-content-between align-items-center mb-3 pb-2 border-bottom" style="border-color: var(--color-border) !important;">
                            <h3 class="h6 mb-0 text-white fw-bold">Item Selecionado</h3>
                            <span class="badge-status badge-success">1x Unidade</span>
                        </div>

                        <div class="d-flex align-items-center gap-3 mb-3">
                            <div id="checkoutProdIcon" style="width: 54px; height: 54px; background: var(--color-muted); border-radius: 10px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;"></div>
                            <div>
                                <h4 class="h6 mb-1 text-white fw-bold" id="checkoutProdNome">Apple iPhone 15 Pro Max 256GB</h4>
                                <span class="text-secondary small" id="checkoutProdSub">Titânio Natural • Tela 6.7 OLED</span>
                            </div>
                        </div>

                        <div class="p-3 rounded mb-3 font-mono small" style="background: var(--color-primary); border: 1px solid var(--color-border);">
                            <div class="d-flex justify-content-between mb-1">
                                <span class="text-secondary">Preço do Produto:</span>
                                <strong class="text-white" id="checkoutProdPreco">R$ 8.999,00</strong>
                            </div>
                            <div class="d-flex justify-content-between mb-1">
                                <span class="text-secondary">Desconto à Vista:</span>
                                <span class="text-success">- R$ 100,00</span>
                            </div>
                            <div class="d-flex justify-content-between mb-1">
                                <span class="text-secondary">Frete com Seguro:</span>
                                <span class="text-success fw-bold">GRÁTIS (Sedex Express)</span>
                            </div>
                            <div class="d-flex justify-content-between pt-2 border-top" style="border-color: var(--color-border) !important;">
                                <strong class="text-white">Total do Pedido:</strong>
                                <strong class="h5 mb-0" id="checkoutProdTotal" style="color: var(--color-accent);">R$ 8.899,00</strong>
                            </div>
                        </div>

                        <!-- Cartão Virtual -->
                        <div class="virtual-card">
                            <div class="d-flex justify-content-between align-items-center mb-3">
                                <div class="chip-sim"></div>
                                <div class="d-flex align-items-center gap-2">
                                    <span class="font-mono small text-secondary">CARTÃO VIRTUAL</span>
                                    <button type="button" class="btn btn-link text-info text-decoration-none p-0 font-mono" onclick="sortearNovoCartao()" title="Sortear novo cartão e titular" style="font-size: 0.72rem;">
                                        ⟳ Sortear
                                    </button>
                                </div>
                            </div>
                            <div class="font-mono fs-5 fw-bold mb-2 text-white" style="letter-spacing: 2px;">
                                •••• •••• •••• <span id="cardUltimosDigitos">7842</span>
                            </div>
                            <div class="d-flex justify-content-between align-items-end font-mono small">
                                <div>
                                    <span class="text-secondary" style="font-size: 0.65rem; display: block;">TITULAR</span>
                                    <span class="text-white fw-bold" id="cardTitularNome">MARIANA R SILVA</span>
                                </div>
                                <div>
                                    <span class="text-secondary" style="font-size: 0.65rem; display: block;">VALIDADE</span>
                                    <span class="text-white" id="cardValidade">11/29</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Coluna 2: A INJEÇÃO DE DADOS (OS BASTIDORES) & FINALIZAR COMPRA -->
                <div class="col-lg-6">
                    <div class="card-custom p-4 mb-4">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <h3 class="h6 mb-0 text-white fw-bold d-flex align-items-center gap-2">
                                <svg class="icon text-info" viewBox="0 0 24 24"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
                                Injeção de Dados (15 Amostras do Dataset)
                            </h3>
                            <span class="badge-pill-custom badge-neutral font-mono" id="badgeModoInjecao">Modo: Sorteio</span>
                        </div>

                        <p class="text-secondary small mb-3">
                            Cada clique envia aos bastidores uma transação com <strong>Time, Amount e V1 a V28</strong> diretamente para o modelo.
                        </p>

                        <!-- Seletor de Modo -->
                        <div class="d-flex gap-2 mb-3">
                            <button
                                type="button"
                                id="btnModoRandom"
                                class="btn-secondary-custom flex-grow-1 small border-success text-white"
                                onclick="setModoInjecao('random')"
                            >
                                🎲 Sorteio Aleatório (Padrão)
                            </button>
                            <button
                                type="button"
                                id="btnModoManual"
                                class="btn-secondary-custom flex-grow-1 small"
                                onclick="setModoInjecao('manual')"
                            >
                                🎯 Seleção Didática Manual
                            </button>
                        </div>

                        <!-- Dropdown de Seleção Manual (Oculto em modo Random) -->
                        <div id="containerSelecaoManual" class="mb-3" style="display: none;">
                            <label for="selectAmostraManual" class="form-label text-secondary small">Escolha a transação de teste:</label>
                            <select
                                id="selectAmostraManual"
                                class="form-select-custom"
                                onchange="selecionarAmostraManual(this.value)"
                            >
                                <!-- Preenchido dinamicamente via JS -->
                            </select>
                        </div>

                        <!-- Cartão da Transação Ativa -->
                        <div class="p-3 rounded border mb-3" style="background: var(--color-primary); border-color: var(--color-border) !important;">
                            <div class="d-flex justify-content-between align-items-center mb-1">
                                <span class="badge-pill-custom badge-neutral font-mono" id="txBadgeId">TX-NORM-001</span>
                                <span class="badge-pill-custom badge-green" id="txBadgeCategoria">Compra Legítima</span>
                            </div>
                            <div class="text-white small fw-bold mb-2" id="txDescricao">Compra Normal #1: Pagamento habitual em horário comercial</div>
                            <div class="d-flex justify-content-between font-mono small text-secondary">
                                <span id="txTimeDisplay">Tempo: 45000s (12:30:00)</span>
                                <span id="txAmountDisplay">Valor Transação: R$ 89,90</span>
                            </div>
                        </div>

                        <!-- Toggle Accordion Payload -->
                        <div class="mb-3">
                            <button
                                type="button"
                                class="btn btn-link text-info text-decoration-none small p-0 font-mono"
                                onclick="togglePayloadAccordion()"
                                id="btnTogglePayload"
                            >
                                ▶ Inspecionar Payload Completo (V1..V28)
                            </button>
                            <pre id="prePayload" class="mt-2 p-2 rounded bg-black small font-mono text-secondary" style="display: none; max-height: 160px; overflow-y: auto; font-size: 0.75rem; border: 1px solid var(--color-border);"></pre>
                        </div>

                        <!-- Botão CTA Principal de Finalizar Compra -->
                        <div class="pt-3 border-top" style="border-color: var(--color-border) !important;">
                            <button id="btnFinalizarCompra" class="btn-cta w-100 py-3" onclick="processarCheckout()">
                                <svg class="icon-lg" viewBox="0 0 24 24"><rect width="20" height="14" x="2" y="5" rx="2"/><line x1="2" x2="22" y1="10" y2="10"/></svg>
                                <span>Finalizar Compra Agora</span>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- ========================================== -->
        <!-- TAB 3: 3. CONCLUSÃO (RESULTADO SE FOI FRAUDE OU NÃO) -->
        <!-- ========================================== -->
        <section id="tab-conclusao" role="tabpanel" style="display: none;">
            <!-- Contêiner do Resultado Preditivo -->
            <div id="conclusaoContainer">
                <div class="card-custom p-5 text-center my-4">
                    <div class="mb-3 text-secondary">
                        <svg class="icon-xl" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 14 14"/></svg>
                    </div>
                    <h3 class="h5 text-white fw-bold mb-2">Nenhuma transação avaliada ainda</h3>
                    <p class="text-secondary small mb-3">Selecione um produto na Vitrine (Aba 1) e finalize a compra no Checkout (Aba 2) para visualizar o parecer antifraude.</p>
                    <button type="button" class="btn-cta" onclick="switchTab('produtos')">
                        <span>Ir para Selecionar Produto &rarr;</span>
                    </button>
                </div>
            </div>

            <!-- Seção Acadêmica e Governança CRISP-DM / Monitoramento de Drift (Slide 18) -->
            <div class="card-custom p-4 mt-4" style="background: rgba(15, 23, 42, 0.5);">
                <div class="d-flex justify-content-between align-items-center" style="cursor: pointer;" onclick="toggleCrispSecao()">
                    <div class="d-flex align-items-center gap-2">
                        <span class="badge-status badge-info">CRISP-DM &amp; XAI</span>
                        <h3 class="h6 mb-0 text-white fw-bold">Explicabilidade do Modelo (XAI), Auditoria do Pipeline &amp; Monitoramento de Data Drift</h3>
                    </div>
                    <span id="crispSecaoStatus" class="text-secondary small font-mono">Expandir Detalhes &darr;</span>
                </div>

                <div id="crispSecaoBody" class="mt-3 pt-3 border-top" style="display: none; border-color: var(--color-border) !important;">
                    <p class="text-secondary small mb-3">
                        Implementação das etapas metodológicas do Projeto Integrador V ministrado pelo <strong>Prof. Dr. Sidney Carlos Ferrari</strong> (FATEC Ourinhos).
                    </p>

                    <!-- ========================================== -->
                    <!-- PAINEL XAI (EXPLAINABLE AI / IA EXPLICÁVEL) -->
                    <!-- ========================================== -->
                    <div class="xai-card p-4 mb-4" id="painelXaiBox">
                        <div class="d-flex flex-wrap justify-content-between align-items-center gap-2 mb-3 pb-2 border-bottom" style="border-color: var(--color-border) !important;">
                            <div class="d-flex align-items-center gap-2">
                                <span class="badge-status badge-info">XAI • IA EXPLICÁVEL</span>
                                <h4 class="h6 mb-0 text-white fw-bold">Interpretabilidade e Causa do Resultado Preditivo</h4>
                            </div>
                            <span class="badge-status badge-neutral font-mono" id="xaiBadgeStatus">Aguardando Avaliação</span>
                        </div>

                        <!-- Caixa de Síntese Explicativa em Linguagem Natural -->
                        <div class="p-3 rounded mb-3" id="xaiNarrativaCard" style="background: var(--color-primary); border-left: 4px solid var(--color-info);">
                            <div class="d-flex align-items-start gap-2">
                                <div class="mt-1" id="xaiIconeNarrativa">
                                    <svg class="icon text-info" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
                                </div>
                                <div class="flex-grow-1">
                                    <div class="small fw-bold text-white mb-1" id="xaiTituloNarrativa">Racional de Decisão do Algoritmo:</div>
                                    <div class="small text-secondary" id="xaiTextoNarrativa">
                                        Nenhuma transação foi processada ainda. Finalize uma compra no <strong>Checkout (Passo 2)</strong> para que o motor XAI analise as variáveis latentes da amostra e gere a árvore de causas que determinou a aprovação, o desafio 2FA ou o bloqueio por fraude.
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Decomposição de Atributos Chave (SHAP/Feature Attribution) -->
                        <div class="mb-3">
                            <div class="d-flex justify-content-between align-items-center mb-2">
                                <span class="small fw-bold text-white font-mono">Contribuição das Variáveis Chave (Top Fatores Determinantes — Célula 41):</span>
                                <span class="text-secondary small font-mono" style="font-size: 0.72rem;">Referência: Z-Score &amp; Correlação Pearson</span>
                            </div>

                            <div class="table-responsive">
                                <table class="table-custom" style="font-size: 0.8rem;">
                                    <thead>
                                        <tr>
                                            <th>Variável</th>
                                            <th>Valor Amostrado</th>
                                            <th>Peso / Importância</th>
                                            <th style="min-width: 140px;">Influência no Score</th>
                                            <th>Interpretação Semântica XAI</th>
                                        </tr>
                                    </thead>
                                    <tbody id="tabelaXaiBody">
                                        <!-- Preenchido dinamicamente via JS com V14, V17, V12, V10, V4, V16, Time, Amount -->
                                    </tbody>
                                </table>
                            </div>
                        </div>

                        <!-- Explicação Metodológica da Engenharia Reversa -->
                        <div class="p-2 rounded font-mono text-secondary small" style="background: rgba(2, 6, 23, 0.6); font-size: 0.72rem;">
                            💡 <strong>Metodologia XAI Aplicada:</strong> As variáveis PCA (V1..V28) foram ranqueadas pelo módulo de sensibilidade baseado nos pesos médios do ensemble de Regressões Logísticas (Célula 41). Variáveis fortemente negativas como <code>V14</code> e <code>V17</code> indicam anomalias drásticas na assinatura do portador, enquanto <code>V16</code> e <code>V25</code> atuam como pesos protetivos de compra rotineira.
                        </div>
                    </div>

                    <div class="row g-3 mb-4">
                        <div class="col-md-4">
                            <div class="card-custom p-3 h-100" style="background: var(--color-muted);">
                                <span class="badge-status badge-info mb-1">Etapas 1 &amp; 2</span>
                                <h4 class="h6 text-white fw-bold">Negócio &amp; Dados</h4>
                                <p class="text-secondary small mb-0">284.807 transações da MegaLoja balanceadas via Under-sampling supervisionado (Cell 41).</p>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card-custom p-3 h-100" style="background: var(--color-muted);">
                                <span class="badge-status badge-success mb-1">Etapas 3 &amp; 4</span>
                                <h4 class="h6 text-white fw-bold">Modelagem &amp; XAI</h4>
                                <p class="text-secondary small mb-0">BaggingClassifier (10 Regressões Logísticas), interpretabilidade e serialização ultracompacta.</p>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card-custom p-3 h-100" style="background: var(--color-muted);">
                                <span class="badge-status badge-warning mb-1">Etapa 5 (Slide 18)</span>
                                <h4 class="h6 text-white fw-bold">Drift &amp; MLOps</h4>
                                <p class="text-secondary small mb-0">Sentinela estatístico autônomo com teste Kolmogorov-Smirnov e cálculo de PSI.</p>
                            </div>
                        </div>
                    </div>

                    <!-- Botão para Teste de Auditoria de Drift ao vivo -->
                    <div class="d-flex flex-wrap justify-content-between align-items-center gap-2 p-3 rounded mb-3" style="background: var(--color-primary); border: 1px solid var(--color-border);">
                        <div>
                            <strong class="text-white small">Executar Teste Estatístico de Deriva (Data Drift):</strong>
                            <div class="text-secondary small">Compara a distribuição em produção com a baseline de treino (KS-test e PSI).</div>
                        </div>
                        <div class="d-flex gap-2">
                            <button type="button" class="btn-secondary-custom btn-sm font-mono" onclick="executarAuditoria(false)">Auditoria Normal</button>
                            <button type="button" class="btn btn-outline-warning btn-sm font-mono" onclick="executarAuditoria(true)">Simular Anomalia</button>
                        </div>
                    </div>

                    <div id="driftLoading" class="text-center py-3" style="display: none;">
                        <div class="spinner-border spinner-border-sm text-info" role="status"></div>
                        <span class="text-secondary small ms-2">Calculando KS & PSI nas variáveis...</span>
                    </div>

                    <div id="driftResultadoCard" class="mt-3" style="display: none;">
                        <div id="alertBoxDrift" class="p-3 rounded border mb-3">
                            <div class="d-flex align-items-center gap-3">
                                <div id="alertDriftIcon"></div>
                                <div>
                                    <h5 id="alertDriftTitle" class="h6 mb-1 fw-bold"></h5>
                                    <p id="alertDriftMsg" class="mb-0 small"></p>
                                </div>
                            </div>
                        </div>
                        <div class="table-responsive">
                            <table class="table-custom">
                                <thead>
                                    <tr>
                                        <th>Variável</th>
                                        <th>Estatística KS</th>
                                        <th>p-value</th>
                                        <th>PSI</th>
                                        <th>Diagnóstico</th>
                                    </tr>
                                </thead>
                                <tbody id="tabelaDriftBody"></tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Footer -->
        <footer class="mt-4 pt-3 border-top text-center text-secondary small" style="border-color: var(--color-border) !important;">
            <p class="mb-0 font-mono" style="font-size: 0.75rem;">MegaLoja S.A. • FATEC Ourinhos • Projeto Integrador V • 2026</p>
        </footer>

    </main>

    <script>
        // Catálogo de Produtos da Loja (Passo 1: Selecionar Produto)
        const PRODUTOS = [
            {
                id: 'prod-1',
                nome: 'Apple iPhone 15 Pro Max 256GB',
                categoria: 'Smartphones',
                subtitulo: 'Titânio Natural • Tela 6.7 OLED • Chip A17 Pro',
                preco: 8999.00,
                icone: '<svg class="icon-xl text-success" viewBox="0 0 24 24"><rect width="14" height="20" x="5" y="2" rx="2"/><line x1="12" y1="18" x2="12.01" y2="18"/></svg>',
                badge: 'MAIS VENDIDO'
            },
            {
                id: 'prod-2',
                nome: 'Notebook Gamer Dell G15',
                categoria: 'Computadores',
                subtitulo: 'Intel i7 • RTX 4050 • 16GB RAM • SSD 512GB',
                preco: 6499.00,
                icone: '<svg class="icon-xl text-info" viewBox="0 0 24 24"><rect width="20" height="14" x="2" y="3" rx="2"/><line x1="2" y1="20" x2="22" y2="20"/></svg>',
                badge: 'ALTA PERFORMANCE'
            },
            {
                id: 'prod-3',
                nome: 'Smart TV Samsung Neo QLED 65 4K',
                categoria: 'TV & Vídeo',
                subtitulo: '120Hz • Mini LED • Neural Quantum 4K',
                preco: 5299.00,
                icone: '<svg class="icon-xl text-warning" viewBox="0 0 24 24"><rect width="20" height="15" x="2" y="3" rx="2"/><polyline points="17 2 12 7 7 2"/></svg>',
                badge: 'CINEMA EM CASA'
            },
            {
                id: 'prod-4',
                nome: 'Console PlayStation 5 Slim 1TB',
                categoria: 'Games',
                subtitulo: 'Controle DualSense • Edição Digital • Áudio 3D',
                preco: 3799.00,
                icone: '<svg class="icon-xl text-primary" style="color: #818cf8 !important;" viewBox="0 0 24 24"><rect width="20" height="12" x="2" y="6" rx="2"/><circle cx="8" cy="12" r="2"/><path d="M15 9h2v6h-2z"/></svg>',
                badge: 'GAMER'
            },
            {
                id: 'prod-5',
                nome: 'Headphone Sony WH-1000XM5 ANC',
                categoria: 'Áudio',
                subtitulo: 'Cancelamento Ativo de Ruído • Bateria 30h',
                preco: 2199.00,
                icone: '<svg class="icon-xl text-info" viewBox="0 0 24 24"><path d="M3 18v-6a9 9 0 0 1 18 0v6"/><path d="M21 19a2 2 0 0 1-2 2h-1a2 2 0 0 1-2-2v-3a2 2 0 0 1 2-2h3zM3 19a2 2 0 0 0 2 2h1a2 2 0 0 0 2-2v-3a2 2 0 0 0-2-2H3z"/></svg>',
                badge: 'OFERTA'
            },
            {
                id: 'prod-6',
                nome: 'Apple Watch Series 9 GPS 45mm',
                categoria: 'Wearables',
                subtitulo: 'Caixa Alumínio • Tela Retina • Sensor ECG',
                preco: 3299.00,
                icone: '<svg class="icon-xl text-success" viewBox="0 0 24 24"><rect width="12" height="16" x="6" y="4" rx="3"/><path d="M9 4V2h6v2M9 20v2h6v-2"/></svg>',
                badge: 'PREMIUM'
            },
            {
                id: 'prod-7',
                nome: 'Monitor LG UltraWide 34 Curvo',
                categoria: 'Monitores',
                subtitulo: '144Hz • 1ms MBR • HDR10 • IPS sRGB 99%',
                preco: 2499.00,
                icone: '<svg class="icon-xl text-warning" viewBox="0 0 24 24"><rect width="20" height="14" x="2" y="3" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>',
                badge: 'PRODUTIVIDADE'
            },
            {
                id: 'prod-8',
                nome: 'Teclado Mecânico Keychron Pro RGB',
                categoria: 'Acessórios',
                subtitulo: 'Switches Hot-swap • Bluetooth 5.1 & USB-C',
                preco: 549.00,
                icone: '<svg class="icon-xl text-white" viewBox="0 0 24 24"><rect width="20" height="12" x="2" y="6" rx="2"/><circle cx="6" cy="10" r="1"/><circle cx="10" cy="10" r="1"/><circle cx="14" cy="10" r="1"/><circle cx="18" cy="10" r="1"/></svg>',
                badge: 'ECONÔMICO'
            }
        ];

        let produtoAtual = PRODUTOS[0];
        let smsTimerInterval = null;
        let smsTimeLeft = 299;
        const EXPECTED_CODE = '849204';

        // Pool de Titulares e Sobrenomes Brasileiros (Isentos dos autores e do professor)
        const NOMES_TITULARES = [
            'MARIANA R SILVA',
            'LUCAS M OLIVEIRA',
            'BEATRIZ C SOUZA',
            'RODRIGO A PEREIRA',
            'JULIANA F SANTOS',
            'GABRIEL H LIMA',
            'CAMILA B CARVALHO',
            'FELIPE N RIBEIRO',
            'LARISSA K ALMEIDA',
            'THIAGO V GOMES',
            'LETICIA D MARTINS',
            'MATHEUS E ROCHA',
            'VANESSA P BARBOSA',
            'BRUNO L CARDOSO',
            'ISABELA T NASCIMENTO',
            'DIEGO S MOREIRA',
            'FERNANDA J CASTRO',
            'RAFAEL O TEIXEIRA',
            'AMANDA V FREITAS',
            'LEONARDO C PINTO',
            'PATRICIA M ANDRADE',
            'GUSTAVO E DIAS',
            'CAROLINA B MOURA',
            'VINICIUS S ARAUJO',
            'RENATA F NOGUEIRA'
        ];

        const VALIDADES_CARTAO = [
            '03/28', '06/28', '09/28', '11/28',
            '02/29', '05/29', '08/29', '11/29',
            '04/30', '07/30', '10/30', '12/30',
            '03/31', '06/31', '09/31', '12/31'
        ];

        let cartaoAtual = {
            titular: 'MARIANA R SILVA',
            final: '7842',
            validade: '11/29'
        };

        function sortearNovoCartao() {
            let novoTitular;
            do {
                const idx = Math.floor(Math.random() * NOMES_TITULARES.length);
                novoTitular = NOMES_TITULARES[idx];
            } while (novoTitular === cartaoAtual.titular && NOMES_TITULARES.length > 1);

            let novoFinal;
            do {
                novoFinal = String(Math.floor(1000 + Math.random() * 9000));
            } while (novoFinal === cartaoAtual.final);

            let novaValidade;
            do {
                const idxV = Math.floor(Math.random() * VALIDADES_CARTAO.length);
                novaValidade = VALIDADES_CARTAO[idxV];
            } while (novaValidade === cartaoAtual.validade && VALIDADES_CARTAO.length > 1);

            cartaoAtual = {
                titular: novoTitular,
                final: novoFinal,
                validade: novaValidade
            };

            const titularEl = document.getElementById('cardTitularNome');
            if (titularEl) titularEl.innerText = cartaoAtual.titular;

            const finalEl = document.getElementById('cardUltimosDigitos');
            if (finalEl) finalEl.innerText = cartaoAtual.final;

            const validadeEl = document.getElementById('cardValidade');
            if (validadeEl) validadeEl.innerText = cartaoAtual.validade;

            return cartaoAtual;
        }

        // Injeção de Dados: 15 Amostras do Dataset Original (7 Normais, 3 Zona Amarela, 5 Fraudes Reais)
        const SAMPLE_TRANSACTIONS = [
            // 7 Compras Normais (Score Baixo < 0.50 -> Check Verde)
            {
                id: "TX-NORM-001",
                description: "Compra Normal #1: Pagamento habitual em horário comercial",
                expected_category: "normal",
                category_label: "Compra Legítima",
                Time: 45000.0, Amount: 89.90,
                V16: 1.6, V25: 1.4, V7: 1.3, V26: 1.1, V12: 1.2, V23: 0.9,
                V15: -0.6, V5: -0.5, V8: -0.6, V21: -0.5
            },
            {
                id: "TX-NORM-002",
                description: "Compra Normal #2: E-commerce recorrente com cliente VIP",
                expected_category: "normal",
                category_label: "Compra Legítima",
                Time: 54000.0, Amount: 149.00,
                V16: 1.8, V25: 1.6, V7: 1.4, V26: 1.2, V12: 1.3, V23: 1.0,
                V15: -0.7, V5: -0.6, V8: -0.7, V21: -0.6
            },
            {
                id: "TX-NORM-003",
                description: "Compra Normal #3: Livro técnico parcelado sem juros",
                expected_category: "normal",
                category_label: "Compra Legítima",
                Time: 62000.0, Amount: 79.50,
                V16: 1.5, V25: 1.3, V7: 1.1, V26: 1.0, V12: 1.1, V23: 0.8,
                V15: -0.5, V5: -0.4, V8: -0.5, V21: -0.4
            },
            {
                id: "TX-NORM-004",
                description: "Compra Normal #4: Periféricos de informática com entrega expressa",
                expected_category: "normal",
                category_label: "Compra Legítima",
                Time: 39000.0, Amount: 249.90,
                V16: 1.4, V25: 1.2, V7: 1.0, V26: 0.9, V12: 1.0, V23: 0.7,
                V15: -0.4, V5: -0.3, V8: -0.4, V21: -0.3
            },
            {
                id: "TX-NORM-005",
                description: "Compra Normal #5: Acessórios para celular na MegaLoja",
                expected_category: "normal",
                category_label: "Compra Legítima",
                Time: 48000.0, Amount: 45.00,
                V16: 1.7, V25: 1.5, V7: 1.3, V26: 1.1, V12: 1.2, V23: 0.9,
                V15: -0.6, V5: -0.5, V8: -0.6, V21: -0.5
            },
            {
                id: "TX-NORM-006",
                description: "Compra Normal #6: Monitor gamer com frete grátis",
                expected_category: "normal",
                category_label: "Compra Legítima",
                Time: 51000.0, Amount: 350.00,
                V16: 1.5, V25: 1.3, V7: 1.2, V26: 1.0, V12: 1.1, V23: 0.8,
                V15: -0.5, V5: -0.4, V8: -0.5, V21: -0.4
            },
            {
                id: "TX-NORM-007",
                description: "Compra Normal #7: Fone Bluetooth com pagamento à vista",
                expected_category: "normal",
                category_label: "Compra Legítima",
                Time: 58000.0, Amount: 199.90,
                V16: 1.9, V25: 1.7, V7: 1.5, V26: 1.3, V12: 1.4, V23: 1.1,
                V15: -0.8, V5: -0.7, V8: -0.8, V21: -0.7
            },
            // 3 Compras na Zona Amarela (Score 0.50 a 0.74 -> Desafio SMS 2FA)
            {
                id: "TX-NORM-008",
                description: "Zona Amarela #1: Compra noturna de celular novo (Requer SMS 2FA)",
                expected_category: "yellow_zone",
                category_label: "Verificação 2FA (SMS)",
                Time: 65000.0, Amount: 380.00,
                V16: 0.78, V25: 0.65, V7: 0.45, V26: 0.40, V12: 0.45, V23: 0.30,
                V15: -0.10, V5: -0.05, V8: -0.10, V21: -0.05
            },
            {
                id: "TX-NORM-009",
                description: "Zona Amarela #2: Novo endereço de entrega cadastrado (Requer SMS 2FA)",
                expected_category: "yellow_zone",
                category_label: "Verificação 2FA (SMS)",
                Time: 65000.0, Amount: 410.00,
                V16: 0.76, V25: 0.65, V7: 0.45, V26: 0.40, V12: 0.45, V23: 0.30,
                V15: -0.10, V5: -0.05, V8: -0.10, V21: -0.05
            },
            {
                id: "TX-NORM-010",
                description: "Zona Amarela #3: Transação acima da média habitual (Requer SMS 2FA)",
                expected_category: "yellow_zone",
                category_label: "Verificação 2FA (SMS)",
                Time: 65000.0, Amount: 450.00,
                V16: 0.75, V25: 0.65, V7: 0.45, V26: 0.40, V12: 0.45, V23: 0.30,
                V15: -0.10, V5: -0.05, V8: -0.10, V21: -0.05
            },
            // 5 Fraudes Reais (Score >= 0.75 -> Bloqueio Imediato: Transação Negada)
            {
                id: "TX-FRAUD-001",
                description: "Fraude #1: Clonagem Noturna Confirmada (Amostra Cell 41)",
                expected_category: "fraud",
                category_label: "Fraude Crítica",
                Time: 406.0, Amount: 149.62,
                V1: -2.31, V2: 1.95, V3: -1.60, V4: 3.99, V5: -0.52, V6: -1.42, V7: -2.53, V8: 1.39,
                V9: -2.77, V10: -2.77, V11: 3.20, V12: -2.89, V13: -0.59, V14: -4.28, V15: 0.38,
                V16: -1.14, V17: -2.83, V18: -0.01, V19: 0.41, V20: 0.12, V21: 0.51, V22: -0.03,
                V23: -0.46, V24: 0.32, V25: 0.04, V26: 0.17, V27: 0.26, V28: -0.14
            },
            {
                id: "TX-FRAUD-002",
                description: "Fraude #2: Compra Abusiva com Cartão Furtado (03:15h)",
                expected_category: "fraud",
                category_label: "Fraude Crítica",
                Time: 11800.0, Amount: 4899.00,
                V1: -3.05, V2: 2.50, V3: -2.90, V4: 4.10, V5: -1.20, V6: -1.80, V7: -3.10, V8: 1.80,
                V9: -3.20, V10: -3.50, V11: 3.80, V12: -3.90, V13: -0.80, V14: -5.10, V15: 0.40,
                V16: -1.80, V17: -3.90, V18: -0.20, V19: 0.60, V20: 0.40, V21: 0.80, V22: -0.10,
                V23: -0.50, V24: 0.20, V25: 0.10, V26: 0.30, V27: 0.40, V28: -0.20
            },
            {
                id: "TX-FRAUD-003",
                description: "Fraude #3: Injeção de Cartão Bloqueado no Emissor",
                expected_category: "fraud",
                category_label: "Fraude Crítica",
                Time: 472.0, Amount: 529.00,
                V1: -2.80, V2: 2.10, V3: -2.20, V4: 3.75, V5: -0.90, V6: -1.60, V7: -2.80, V8: 1.50,
                V9: -2.95, V10: -3.10, V11: 3.40, V12: -3.20, V13: -0.70, V14: -4.60, V15: 0.25,
                V16: -1.40, V17: -3.20, V18: -0.10, V19: 0.50, V20: 0.25, V21: 0.65, V22: -0.05,
                V23: -0.48, V24: 0.25, V25: 0.08, V26: 0.22, V27: 0.32, V28: -0.18
            },
            {
                id: "TX-FRAUD-004",
                description: "Fraude #4: Ataque Automatizado de Teste de Cartões (Botnet)",
                expected_category: "fraud",
                category_label: "Fraude Crítica",
                Time: 8500.0, Amount: 1290.00,
                V1: -2.55, V2: 2.05, V3: -1.95, V4: 3.85, V5: -0.75, V6: -1.50, V7: -2.65, V8: 1.45,
                V9: -2.85, V10: -2.95, V11: 3.30, V12: -3.05, V13: -0.65, V14: -4.45, V15: 0.32,
                V16: -1.25, V17: -3.05, V18: -0.05, V19: 0.45, V20: 0.18, V21: 0.58, V22: -0.04,
                V23: -0.47, V24: 0.28, V25: 0.06, V26: 0.19, V27: 0.29, V28: -0.16
            },
            {
                id: "TX-FRAUD-005",
                description: "Fraude #5: Golpe de Alto Valor via Proxy Anônimo",
                expected_category: "fraud",
                category_label: "Fraude Crítica",
                Time: 24000.0, Amount: 8999.00,
                V1: -3.40, V2: 2.80, V3: -3.10, V4: 4.50, V5: -1.50, V6: -2.00, V7: -3.40, V8: 2.00,
                V9: -3.50, V10: -3.80, V11: 4.10, V12: -4.20, V13: -0.90, V14: -5.50, V15: 0.45,
                V16: -2.00, V17: -4.20, V18: -0.30, V19: 0.70, V20: 0.50, V21: 0.90, V22: -0.15,
                V23: -0.55, V24: 0.15, V25: 0.12, V26: 0.35, V27: 0.45, V28: -0.25
            }
        ];

        let modoInjecao = 'random'; // 'random' | 'manual'
        let selectedTxId = SAMPLE_TRANSACTIONS[0].id;
        let activeTx = SAMPLE_TRANSACTIONS[0];
        let showPayload = false;
        let currentPCA = { ...SAMPLE_TRANSACTIONS[0] };

        function formatarBRL(val) {
            return Number(val).toLocaleString('pt-BR', { style: 'currency', currency: 'BRL' });
        }

        function formatarTempoMMSS(secs) {
            const m = Math.floor(Math.max(0, secs) / 60);
            const s = Math.max(0, secs) % 60;
            return String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
        }

        // Renderiza catálogo de produtos da Vitrine
        function renderizarCatalogo() {
            const container = document.getElementById('catalogoProdutosContainer');
            if (!container) return;
            container.innerHTML = '';
            PRODUTOS.forEach(p => {
                const col = document.createElement('div');
                col.className = 'col-sm-6 col-lg-3';
                col.innerHTML = `
                    <div class="product-card">
                        <div>
                            <div class="d-flex justify-content-between align-items-center mb-2">
                                <span class="badge-status badge-info" style="font-size: 0.68rem;">${p.categoria}</span>
                                <span class="badge-status badge-neutral" style="font-size: 0.65rem;">${p.badge}</span>
                            </div>
                            <div class="product-img-box">
                                ${p.icone}
                            </div>
                            <h3 class="h6 text-white fw-bold mb-1">${p.nome}</h3>
                            <p class="text-secondary small mb-3" style="font-size: 0.78rem; min-height: 38px;">${p.subtitulo}</p>
                        </div>
                        <div class="pt-2 border-top" style="border-color: var(--color-border) !important;">
                            <div class="text-secondary small" style="font-size: 0.75rem;">À vista ou 10x sem juros:</div>
                            <div class="h5 mb-2 font-mono text-white fw-bold">${formatarBRL(p.preco)}</div>
                            <button type="button" class="btn-cta btn-sm w-100" onclick="selecionarProduto('${p.id}')">
                                <span>Selecionar &rarr;</span>
                            </button>
                        </div>
                    </div>
                `;
                container.appendChild(col);
            });
        }

        // Seleciona um produto e vai para a aba 2 (Checkout)
        function selecionarProduto(id) {
            const p = PRODUTOS.find(item => item.id === id);
            if (!p) return;
            produtoAtual = p;

            // Sorteia novo titular e final de cartão a cada nova compra (sempre diferente da anterior)
            sortearNovoCartao();

            // Atualiza resumo no Checkout
            const iconEl = document.getElementById('checkoutProdIcon');
            if (iconEl) iconEl.innerHTML = p.icone;
            const nomeEl = document.getElementById('checkoutProdNome');
            if (nomeEl) nomeEl.innerText = p.nome;
            const subEl = document.getElementById('checkoutProdSub');
            if (subEl) subEl.innerText = p.subtitulo;
            const precoEl = document.getElementById('checkoutProdPreco');
            if (precoEl) precoEl.innerText = formatarBRL(p.preco);

            const total = Math.max(10, p.preco - 100.0);
            const totalEl = document.getElementById('checkoutProdTotal');
            if (totalEl) totalEl.innerText = formatarBRL(total);
            const amountEl = document.getElementById('amount_val');
            if (amountEl) amountEl.value = currentPCA.Amount.toFixed(2);

            switchTab('checkout');
        }

        // Navegação de Abas
        function switchTab(tabName) {
            const tabP = document.getElementById('tab-produtos');
            const tabC = document.getElementById('tab-checkout');
            const tabRes = document.getElementById('tab-conclusao');
            if (tabP) tabP.style.display = tabName === 'produtos' ? 'block' : 'none';
            if (tabC) tabC.style.display = tabName === 'checkout' ? 'block' : 'none';
            if (tabRes) tabRes.style.display = tabName === 'conclusao' ? 'block' : 'none';

            const btnP = document.getElementById('tab-btn-produtos');
            const btnC = document.getElementById('tab-btn-checkout');
            const btnRes = document.getElementById('tab-btn-conclusao');
            if (btnP) btnP.classList.toggle('active', tabName === 'produtos');
            if (btnC) btnC.classList.toggle('active', tabName === 'checkout');
            if (btnRes) btnRes.classList.toggle('active', tabName === 'conclusao');

            window.scrollTo({ top: 0, behavior: 'smooth' });
        }

        // Processa o Checkout e vai para a aba 3 (Conclusão)
        async function processarCheckout() {
            const btn = document.getElementById('btnFinalizarCompra');
            if (btn) {
                btn.disabled = true;
                btn.innerHTML = `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Avaliando Risco na Nuvem Vercel...`;
            }

            // 1. Sorteio ou Seleção Manual da Transação
            let currentTx;
            if (modoInjecao === 'random') {
                const randomIndex = Math.floor(Math.random() * SAMPLE_TRANSACTIONS.length);
                currentTx = SAMPLE_TRANSACTIONS[randomIndex];
                selectedTxId = currentTx.id;
                const sel = document.getElementById('selectAmostraManual');
                if (sel) sel.value = currentTx.id;
            } else {
                currentTx = SAMPLE_TRANSACTIONS.find(t => t.id === selectedTxId) || SAMPLE_TRANSACTIONS[0];
            }
            activeTx = currentTx;
            atualizarCardAmostra(activeTx);

            // 2. Prepara o Payload completo (Time, Amount, V1 a V28)
            const payload = {
                Time: currentTx.Time,
                Amount: currentTx.Amount
            };
            for (let i = 1; i <= 28; i++) {
                payload['V' + i] = currentTx['V' + i] !== undefined ? currentTx['V' + i] : 0.0;
            }
            currentPCA = { ...payload };

            const t0 = performance.now();
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await response.json();
                const t1 = performance.now();
                const latencyMs = Math.round(t1 - t0);

                // Renderiza o resultado na aba 3 (Conclusão)
                renderizarConclusao(data, latencyMs);
                switchTab('conclusao');
            } catch (err) {
                alert("Erro ao consultar o modelo de Machine Learning: " + err);
            } finally {
                if (btn) {
                    btn.disabled = false;
                    btn.innerHTML = `<svg class="icon-lg" viewBox="0 0 24 24"><rect width="20" height="14" x="2" y="5" rx="2"/><line x1="2" x2="22" y1="10" y2="10"/></svg> <span>Finalizar Compra Agora</span>`;
                }
            }
        }

        // Renderiza o resultado na Aba 3 (Conclusão): Se foi fraude ou não!
        function renderizarConclusao(data, latencyMs) {
            const container = document.getElementById('conclusaoContainer');
            if (smsTimerInterval) clearInterval(smsTimerInterval);

            const probPercent = (data.fraud_probability * 100).toFixed(2);
            const totalFormatado = formatarBRL(currentPCA.Amount);

            // CENÁRIO 1: FRAUDE CONFIRMADA (Score >= 0.75) -> AVISO VERMELHO
            if (data.fraud_probability >= 0.75) {
                container.innerHTML = `
                    <div class="card-custom p-4 p-md-5 text-center border" style="background: rgba(239, 68, 68, 0.08); border-color: var(--color-destructive) !important; box-shadow: var(--shadow-glow-danger);">
                        <div class="mb-3" style="color: var(--color-destructive);">
                            <svg class="icon-xl" viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                        </div>
                        <span class="badge-status badge-danger mb-2">BLOQUEIO PREVENTIVO IMEDIATO • FRAUDE DETECTADA</span>
                        <h2 class="h3 text-danger fw-bold mb-2">Transação Negada por Política de Segurança</h2>
                        <p class="text-secondary small mb-4" style="max-width: 650px; margin: 0 auto;">
                            O motor antifraude identificou desvios comportamentais severos compatíveis com clonagem e ataque automatizado de cartões. A compra de <strong>${produtoAtual.nome}</strong> (${totalFormatado}) para o titular <strong>${cartaoAtual.titular}</strong> (Cartão •••• ${cartaoAtual.final}) foi barrada em milissegundos.
                        </p>

                        <!-- Tabela de Auditoria do Bloqueio -->
                        <div class="p-3 rounded mb-4 text-start font-mono small mx-auto" style="max-width: 650px; background: var(--color-primary); border: 1px solid var(--color-border);">
                            <div class="d-flex justify-content-between mb-1">
                                <span class="text-secondary">Parecer do Modelo:</span>
                                <strong class="text-danger">FRAUDE DETECTADA</strong>
                            </div>
                            <div class="d-flex justify-content-between mb-1">
                                <span class="text-secondary">Amostra do Dataset:</span>
                                <span class="text-white">${activeTx ? activeTx.id + ' (' + activeTx.category_label + ')' : 'TX-DATASET'}</span>
                            </div>
                            <div class="d-flex justify-content-between mb-1">
                                <span class="text-secondary">Cartão Monitorado:</span>
                                <span class="text-danger">•••• •••• •••• ${cartaoAtual.final} (${cartaoAtual.titular})</span>
                            </div>
                            <div class="d-flex justify-content-between mb-1">
                                <span class="text-secondary">Probabilidade Calculada:</span>
                                <strong class="text-danger">${probPercent}%</strong>
                            </div>
                            <div class="d-flex justify-content-between mb-1">
                                <span class="text-secondary">Faixa de Risco:</span>
                                <span class="text-danger">CRÍTICO (&ge; 75%)</span>
                            </div>
                            <div class="d-flex justify-content-between mb-1">
                                <span class="text-secondary">Algoritmo em Operação:</span>
                                <span class="text-white">${data.model_name || 'BaggingClassifier (RL)'}</span>
                            </div>
                            <div class="d-flex justify-content-between mb-1">
                                <span class="text-secondary">Latência da Inferência:</span>
                                <span class="text-white">${latencyMs} ms</span>
                            </div>
                            <div class="d-flex justify-content-between">
                                <span class="text-secondary">Ação Automatizada:</span>
                                <span class="text-white">Bloqueio Imediato &bull; Alerta de Segurança SIEM</span>
                            </div>
                        </div>

                        <div class="d-flex flex-wrap justify-content-center gap-3">
                            <button type="button" class="btn-cta" onclick="switchTab('produtos')">
                                <svg class="icon" viewBox="0 0 24 24"><path d="M6 2 3 6v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6l-3-4Z"/><path d="M3 6h18"/><path d="M16 10a4 4 0 0 1-8 0"/></svg>
                                <span>Escolher Outro Produto (Voltar à Loja)</span>
                            </button>
                            <button type="button" class="btn-secondary-custom" onclick="switchTab('checkout')">
                                <span>Alterar Cenário no Checkout</span>
                            </button>
                            <button type="button" class="btn btn-outline-danger font-mono" onclick="abrirEVisualizarXAI()">
                                🔍 Explicabilidade XAI da Fraude &darr;
                            </button>
                        </div>
                    </div>
                `;
                atualizarPainelXAI(activeTx, data);
                return;
            }

            // CENÁRIO 2: ZONA AMARELA / RISCO MÉDIO (50% a 74%) -> DESAFIO 2FA SMS COM TEMPORIZADOR
            if (data.fraud_probability >= 0.50) {
                smsTimeLeft = 299; // 04:59

                container.innerHTML = `
                    <div class="card-custom p-4 p-md-5 text-center border" id="card2fa" style="background: rgba(245, 158, 11, 0.08); border-color: var(--color-warning) !important; box-shadow: var(--shadow-glow-warning);">
                        <div class="mb-3" style="color: var(--color-warning);">
                            <svg class="icon-xl" viewBox="0 0 24 24"><rect width="14" height="20" x="5" y="2" rx="2"/><line x1="12" y1="18" x2="12.01" y2="18"/></svg>
                        </div>
                        <span class="badge-status badge-warning mb-2">AUTENTICAÇÃO EM 2 ETAPAS (2FA) • ZONA AMARELA</span>
                        <h2 class="h4 text-white fw-bold mb-2">Atenção: Insira o código SMS enviado ao seu celular</h2>
                        <p class="text-secondary small mb-3" style="max-width: 600px; margin: 0 auto;">
                            Identificamos um padrão de compra atípico para o titular <strong>${cartaoAtual.titular}</strong> (Cartão final <strong>•••• ${cartaoAtual.final}</strong>). Enviamos um código de 6 dígitos para o celular cadastrado com final <strong>(11) 98***-**${cartaoAtual.final.slice(-2)}</strong>.
                        </p>
                        <div class="text-warning small font-mono mb-3" style="font-size: 0.8rem;">
                            ⚠️ Qualquer código diferente do emitido ou o término do temporizador bloqueará a transação como fraude.
                        </div>

                        <!-- Formulário 2FA -->
                        <div class="p-4 rounded mx-auto text-start" style="max-width: 480px; background: var(--color-primary); border: 1px solid var(--color-border);">
                            <form onsubmit="confirmarSmsConclusao(event)">
                                <div class="mb-3 text-center">
                                    <label class="form-label text-secondary small font-mono mb-2">Código SMS de Verificação:</label>
                                    <input
                                        type="text"
                                        id="inputSmsConclusao"
                                        class="form-control-custom text-center fw-bold fs-4 py-2"
                                        placeholder="Ex: 849204"
                                        maxlength="6"
                                        autofocus
                                    />
                                </div>

                                <div class="d-flex justify-content-between align-items-center mb-2">
                                    <button type="button" class="btn btn-link text-info text-decoration-none small p-0 font-mono" onclick="document.getElementById('inputSmsConclusao').value='849204'">
                                        Preencher Código de Teste (849204)
                                    </button>
                                    <span class="small font-mono text-warning" id="smsTimerDisplay">
                                        ⏱️ Expira em 04:59
                                    </span>
                                </div>

                                <!-- Botão didático para forçar o timeout -->
                                <div class="d-flex justify-content-between align-items-center mb-3 p-2 rounded" style="background: rgba(239, 68, 68, 0.08); border: 1px dashed rgba(239, 68, 68, 0.3);">
                                    <span class="text-secondary small font-mono" style="font-size: 0.74rem;">Demonstração para o Professor:</span>
                                    <button type="button" class="btn btn-sm btn-outline-danger font-mono py-0 px-2" style="font-size: 0.75rem;" onclick="simularTimeoutConclusao()">
                                        ⏱️ Simular Timeout (0s)
                                    </button>
                                </div>

                                <div class="d-flex gap-2 mb-2">
                                    <button type="button" class="btn-secondary-custom flex-grow-1" onclick="switchTab('checkout')">
                                        Voltar ao Checkout
                                    </button>
                                    <button type="submit" class="btn-cta flex-grow-1" style="background: var(--color-warning); color: #000;">
                                        Confirmar Código SMS
                                    </button>
                                </div>

                                <div class="text-center pt-2 border-top" style="border-color: var(--color-border) !important;">
                                    <button type="button" class="btn btn-link text-info text-decoration-none small font-mono p-0" onclick="abrirEVisualizarXAI()">
                                        🔍 Por que esta transação exigiu 2FA? (Ver XAI &darr;)
                                    </button>
                                </div>
                            </form>
                        </div>
                    </div>
                `;
                atualizarPainelXAI(activeTx, data);

                // Inicia o temporizador ativo regressivo
                smsTimerInterval = setInterval(() => {
                    smsTimeLeft--;
                    const el = document.getElementById('smsTimerDisplay');
                    if (el) {
                        el.innerText = '⏱️ Expira em ' + formatarTempoMMSS(smsTimeLeft);
                        if (smsTimeLeft <= 60) el.className = "small font-mono text-danger fw-bold";
                    }
                    if (smsTimeLeft <= 0) {
                        clearInterval(smsTimerInterval);
                        rejeitarFraudeConclusao("Tempo limite do código SMS expirado (00:00). Transação bloqueada preventivamente pela política de segurança.");
                    }
                }, 1000);

                return;
            }

            // CENÁRIO 3: SCORE BAIXO (< 50%) -> PAGAMENTO APROVADO COM CHECK VERDE
            container.innerHTML = `
                <div class="card-custom p-4 p-md-5 text-center border" style="background: rgba(34, 197, 94, 0.08); border-color: var(--color-accent) !important; box-shadow: var(--shadow-glow-accent);">
                    <div class="mb-3" style="color: var(--color-accent);">
                        <svg class="icon-xl" viewBox="0 0 24 24"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                    </div>
                    <span class="badge-status badge-success mb-2">SCORE SEGURO • AUTORIZADO</span>
                    <h2 class="h3 text-success fw-bold mb-2">Pagamento Aprovado com Sucesso!</h2>
                    <p class="text-secondary small mb-4" style="max-width: 650px; margin: 0 auto;">
                        Transação autorizada com sucesso pelo emissor para o titular <strong>${cartaoAtual.titular}</strong> (Cartão •••• ${cartaoAtual.final}). O pedido de <strong>${produtoAtual.nome}</strong> no valor de <strong>${totalFormatado}</strong> foi confirmado e liberado para expedição imediata!
                    </p>

                    <!-- Tabela de Auditoria do Pedido Aprovado -->
                    <div class="p-3 rounded mb-4 text-start font-mono small mx-auto" style="max-width: 650px; background: var(--color-primary); border: 1px solid var(--color-border);">
                        <div class="d-flex justify-content-between mb-1">
                            <span class="text-secondary">Status da Transação:</span>
                            <strong class="text-success">APROVADA (NÃO É FRAUDE)</strong>
                        </div>
                        <div class="d-flex justify-content-between mb-1">
                            <span class="text-secondary">Amostra do Dataset:</span>
                            <span class="text-white">${activeTx ? activeTx.id + ' (' + activeTx.category_label + ')' : 'TX-DATASET'}</span>
                        </div>
                        <div class="d-flex justify-content-between mb-1">
                            <span class="text-secondary">Titular & Cartão:</span>
                            <span class="text-white">${cartaoAtual.titular} (•••• ${cartaoAtual.final})</span>
                        </div>
                        <div class="d-flex justify-content-between mb-1">
                            <span class="text-secondary">Probabilidade de Fraude:</span>
                            <strong class="text-success">${probPercent}%</strong>
                        </div>
                        <div class="d-flex justify-content-between mb-1">
                            <span class="text-secondary">Limiar de Segurança:</span>
                            <span class="text-white">50.00% (Abaixo do threshold)</span>
                        </div>
                        <div class="d-flex justify-content-between mb-1">
                            <span class="text-secondary">Modelo Antifraude:</span>
                            <span class="text-white">${data.model_name || 'BaggingClassifier (RL)'}</span>
                        </div>
                        <div class="d-flex justify-content-between mb-1">
                            <span class="text-secondary">Tempo de Resposta:</span>
                            <span class="text-white">${latencyMs} ms (Vercel Serverless)</span>
                        </div>
                        <div class="d-flex justify-content-between">
                            <span class="text-secondary">Código de Rastreio:</span>
                            <span class="text-white">#ML-20261111-SEDEX</span>
                        </div>
                    </div>

                    <div class="d-flex flex-wrap justify-content-center gap-3">
                        <button type="button" class="btn-cta" onclick="switchTab('produtos')">
                            <svg class="icon" viewBox="0 0 24 24"><path d="M6 2 3 6v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6l-3-4Z"/><path d="M3 6h18"/><path d="M16 10a4 4 0 0 1-8 0"/></svg>
                            <span>Comprar Outro Produto (Voltar à Loja)</span>
                        </button>
                        <button type="button" class="btn-secondary-custom" onclick="switchTab('checkout')">
                            <span>Ver Detalhes do Checkout</span>
                        </button>
                        <button type="button" class="btn btn-outline-success font-mono" onclick="abrirEVisualizarXAI()">
                            🔍 Inspecionar Explicabilidade XAI &darr;
                        </button>
                    </div>
                </div>
            `;
            atualizarPainelXAI(activeTx, data);
        }

        // Validação estrita do código SMS na Conclusão
        function confirmarSmsConclusao(e) {
            if (e) e.preventDefault();
            const input = document.getElementById('inputSmsConclusao');
            const code = (input ? input.value : "").trim();

            if (smsTimerInterval) clearInterval(smsTimerInterval);

            if (code === EXPECTED_CODE) {
                // APROVAÇÃO COM SUCESSO!
                const container = document.getElementById('conclusaoContainer');
                const totalFormatado = formatarBRL(currentPCA.Amount);
                container.innerHTML = `
                    <div class="card-custom p-4 p-md-5 text-center border" style="background: rgba(34, 197, 94, 0.08); border-color: var(--color-accent) !important; box-shadow: var(--shadow-glow-accent);">
                        <div class="mb-3" style="color: var(--color-accent);">
                            <svg class="icon-xl" viewBox="0 0 24 24"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                        </div>
                        <span class="badge-status badge-success mb-2">AUTENTICAÇÃO 2FA VALIDADA COM SUCESSO</span>
                        <h2 class="h3 text-success fw-bold mb-2">Pagamento Aprovado!</h2>
                        <p class="text-secondary small mb-4" style="max-width: 650px; margin: 0 auto;">
                            Identidade do titular <strong>${cartaoAtual.titular}</strong> (Cartão final •••• ${cartaoAtual.final}) verificada com sucesso via token SMS 2FA. O pedido de <strong>${produtoAtual.nome}</strong> no valor de <strong>${totalFormatado}</strong> foi liberado para envio imediato!
                        </p>

                        <div class="p-3 rounded mb-4 text-start font-mono small mx-auto" style="max-width: 650px; background: var(--color-primary); border: 1px solid var(--color-border);">
                            <div class="d-flex justify-content-between mb-1">
                                <span class="text-secondary">Status Final:</span>
                                <strong class="text-success">LIBERADO APÓS 2FA</strong>
                            </div>
                            <div class="d-flex justify-content-between mb-1">
                                <span class="text-secondary">Token Validado:</span>
                                <span class="text-white">849204 (SMS Seguro)</span>
                            </div>
                            <div class="d-flex justify-content-between">
                                <span class="text-secondary">Expedição:</span>
                                <span class="text-white">Emissão de Nota Fiscal e Coleta Sedex</span>
                            </div>
                        </div>

                        <div class="d-flex flex-wrap justify-content-center gap-3">
                            <button type="button" class="btn-cta" onclick="switchTab('produtos')">
                                <svg class="icon" viewBox="0 0 24 24"><path d="M6 2 3 6v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6l-3-4Z"/><path d="M3 6h18"/><path d="M16 10a4 4 0 0 1-8 0"/></svg>
                                <span>Comprar Outro Produto (Voltar à Loja)</span>
                            </button>
                            <button type="button" class="btn-secondary-custom" onclick="switchTab('checkout')">
                                <span>Voltar ao Checkout</span>
                            </button>
                            <button type="button" class="btn btn-outline-success font-mono" onclick="abrirEVisualizarXAI()">
                                🔍 Inspecionar Explicabilidade XAI &darr;
                            </button>
                        </div>
                    </div>
                `;
                atualizarPainelXAI(activeTx, { fraud_probability: 0.05, model_name: 'BaggingClassifier (RL)' });
            } else {
                // QUALQUER OUTRO CÓDIGO INCORRETO -> FRAUDE IMEDIATA!
                rejeitarFraudeConclusao('Tentativa de Fraude Detectada: Código SMS digitado (' + (code || 'vazio') + ') é inválido e não confere com o token emitido.');
            }
        }

        function simularTimeoutConclusao() {
            smsTimeLeft = 0;
            if (smsTimerInterval) clearInterval(smsTimerInterval);
            rejeitarFraudeConclusao("Tempo limite do código SMS expirado (00:00). Transação bloqueada preventivamente pela política de segurança.");
        }

        function rejeitarFraudeConclusao(motivo) {
            if (smsTimerInterval) clearInterval(smsTimerInterval);
            const container = document.getElementById('conclusaoContainer');
            const totalFormatado = formatarBRL(currentPCA.Amount);

            container.innerHTML = `
                <div class="card-custom p-4 p-md-5 text-center border" style="background: rgba(239, 68, 68, 0.08); border-color: var(--color-destructive) !important; box-shadow: var(--shadow-glow-danger);">
                    <div class="mb-3" style="color: var(--color-destructive);">
                        <svg class="icon-xl" viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                    </div>
                    <span class="badge-status badge-danger mb-2">FALHA NO DESAFIO 2FA • BLOQUEIO PREVENTIVO</span>
                    <h2 class="h3 text-danger fw-bold mb-2">Transação Negada por Política de Segurança</h2>
                    <p class="text-secondary small mb-4" style="max-width: 650px; margin: 0 auto;">
                        ${motivo} A compra do produto <strong>${produtoAtual.nome}</strong> (${totalFormatado}) foi recusada e o cartão final <strong>${cartaoAtual.final}</strong> do titular <strong>${cartaoAtual.titular}</strong> foi preventivamente bloqueado.
                    </p>

                    <div class="p-3 rounded mb-4 text-start font-mono small mx-auto" style="max-width: 650px; background: var(--color-primary); border: 1px solid var(--color-border);">
                        <div class="d-flex justify-content-between mb-1">
                            <span class="text-secondary">Diagnóstico Antifraude:</span>
                            <strong class="text-danger">REJEITADO (FALHA DE AUTENTICAÇÃO)</strong>
                        </div>
                        <div class="d-flex justify-content-between mb-1">
                            <span class="text-secondary">Classificação de Risco:</span>
                            <span class="text-danger">CRÍTICO (Tentativa de Invasão de Conta)</span>
                        </div>
                        <div class="d-flex justify-content-between">
                            <span class="text-secondary">Decisão Automatizada:</span>
                            <span class="text-white">Bloqueio Imediato &bull; Alerta SIEM Disparado</span>
                        </div>
                    </div>

                    <div class="d-flex flex-wrap justify-content-center gap-3">
                        <button type="button" class="btn-cta" onclick="switchTab('produtos')">
                            <svg class="icon" viewBox="0 0 24 24"><path d="M6 2 3 6v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6l-3-4Z"/><path d="M3 6h18"/><path d="M16 10a4 4 0 0 1-8 0"/></svg>
                            <span>Escolher Outro Produto (Voltar à Loja)</span>
                        </button>
                        <button type="button" class="btn-secondary-custom" onclick="switchTab('checkout')">
                            <span>Tentar Outro Cenário no Checkout</span>
                        </button>
                        <button type="button" class="btn btn-outline-danger font-mono" onclick="abrirEVisualizarXAI()">
                            🔍 Explicabilidade XAI do Bloqueio &darr;
                        </button>
                    </div>
                </div>
            `;
            atualizarPainelXAI(activeTx, { fraud_probability: 0.999, model_name: 'BaggingClassifier (RL)' });
        }

        // Funções da Injeção de Dados (15 Amostras do Dataset)
        function setModoInjecao(modo) {
            modoInjecao = modo;
            const btnRandom = document.getElementById('btnModoRandom');
            const btnManual = document.getElementById('btnModoManual');
            const badge = document.getElementById('badgeModoInjecao');
            const containerManual = document.getElementById('containerSelecaoManual');

            if (modo === 'random') {
                if (btnRandom) btnRandom.className = 'btn-secondary-custom flex-grow-1 small border-success text-white';
                if (btnManual) btnManual.className = 'btn-secondary-custom flex-grow-1 small';
                if (badge) badge.innerText = 'Modo: Sorteio';
                if (containerManual) containerManual.style.display = 'none';
            } else {
                if (btnRandom) btnRandom.className = 'btn-secondary-custom flex-grow-1 small';
                if (btnManual) btnManual.className = 'btn-secondary-custom flex-grow-1 small border-info text-white';
                if (badge) badge.innerText = 'Modo: Manual';
                if (containerManual) containerManual.style.display = 'block';
            }
        }

        function popularSelectManual() {
            const sel = document.getElementById('selectAmostraManual');
            if (!sel) return;
            const normais = SAMPLE_TRANSACTIONS.filter(t => t.expected_category === 'normal');
            const suspeitas = SAMPLE_TRANSACTIONS.filter(t => t.expected_category === 'yellow_zone');
            const fraudes = SAMPLE_TRANSACTIONS.filter(t => t.expected_category === 'fraud');

            sel.innerHTML = `
                <optgroup label="🟢 Compras Normais (Score Baixo -> Aprovado)">
                    ${normais.map(t => `<option value="${t.id}">${t.id} — ${t.description}</option>`).join('')}
                </optgroup>
                <optgroup label="🟡 Zona Amarela (Score Médio -> Desafio SMS 2FA)">
                    ${suspeitas.map(t => `<option value="${t.id}">${t.id} — ${t.description}</option>`).join('')}
                </optgroup>
                <optgroup label="🔴 Fraudes Reais (Score Alto -> Bloqueio Imediato)">
                    ${fraudes.map(t => `<option value="${t.id}">${t.id} — ${t.description}</option>`).join('')}
                </optgroup>
            `;
            sel.value = selectedTxId;
        }

        function selecionarAmostraManual(id) {
            selectedTxId = id;
            const found = SAMPLE_TRANSACTIONS.find(t => t.id === id);
            if (found) {
                activeTx = found;
                currentPCA = { ...found };
                atualizarCardAmostra(activeTx);
                atualizarPainelXAI(activeTx, null);
            }
        }

        function atualizarCardAmostra(tx) {
            if (!tx) return;
            const badgeId = document.getElementById('txBadgeId');
            if (badgeId) badgeId.innerText = tx.id;

            const badgeCat = document.getElementById('txBadgeCategoria');
            if (badgeCat) {
                badgeCat.innerText = tx.category_label;
                badgeCat.className = 'badge-pill-custom ' + (
                    tx.expected_category === 'fraud' ? 'badge-red' :
                    tx.expected_category === 'yellow_zone' ? 'badge-yellow' : 'badge-green'
                );
            }

            const desc = document.getElementById('txDescricao');
            if (desc) desc.innerText = tx.description;

            const timeDisp = document.getElementById('txTimeDisplay');
            if (timeDisp) {
                const h = Math.floor(tx.Time / 3600) % 24;
                const m = Math.floor((tx.Time % 3600) / 60);
                const s = Math.floor(tx.Time % 60);
                const pad = (n) => String(n).padStart(2, '0');
                timeDisp.innerText = `Tempo: ${tx.Time}s (${pad(h)}:${pad(m)}:${pad(s)})`;
            }

            const amountDisp = document.getElementById('txAmountDisplay');
            if (amountDisp) {
                amountDisp.innerText = `Valor Transação: ${formatarBRL(tx.Amount)}`;
            }

            const pre = document.getElementById('prePayload');
            if (pre) {
                pre.innerText = JSON.stringify(tx, null, 2);
            }
        }

        function togglePayloadAccordion() {
            showPayload = !showPayload;
            const pre = document.getElementById('prePayload');
            const btn = document.getElementById('btnTogglePayload');
            if (pre) pre.style.display = showPayload ? 'block' : 'none';
            if (btn) btn.innerText = showPayload ? '▼ Ocultar Componentes PCA (V1..V28)' : '▶ Inspecionar Payload Completo (V1..V28)';
            if (showPayload && pre) {
                pre.innerText = JSON.stringify(activeTx, null, 2);
            }
        }

        function toggleCrispSecao() {
            const body = document.getElementById('crispSecaoBody');
            const status = document.getElementById('crispSecaoStatus');
            if (!body || !status) return;
            if (body.style.display === 'none') {
                body.style.display = 'block';
                status.innerHTML = 'Recolher Detalhes &uarr;';
            } else {
                body.style.display = 'none';
                status.innerHTML = 'Expandir Detalhes &darr;';
            }
        }

        function abrirEVisualizarXAI() {
            const body = document.getElementById('crispSecaoBody');
            const status = document.getElementById('crispSecaoStatus');
            if (body) body.style.display = 'block';
            if (status) status.innerHTML = 'Recolher Detalhes &uarr;';
            const xaiBox = document.getElementById('painelXaiBox');
            if (xaiBox) {
                xaiBox.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }

        function atualizarPainelXAI(tx, result) {
            const targetTx = tx || activeTx || (SAMPLE_TRANSACTIONS && SAMPLE_TRANSACTIONS[0]);
            if (!targetTx) return;

            const badgeStatus = document.getElementById('xaiBadgeStatus');
            const cardNarrativa = document.getElementById('xaiNarrativaCard');
            const iconeNarrativa = document.getElementById('xaiIconeNarrativa');
            const tituloNarrativa = document.getElementById('xaiTituloNarrativa');
            const textoNarrativa = document.getElementById('xaiTextoNarrativa');
            const tabelaBody = document.getElementById('tabelaXaiBody');

            const hora = Math.floor(targetTx.Time / 3600) % 24;
            const horaFormatada = String(hora).padStart(2, '0') + 'h';
            const totalFormatado = formatarBRL(targetTx.Amount);

            if (!result) {
                // Estado de espera / pré-inferência
                if (badgeStatus) {
                    badgeStatus.className = 'badge-status badge-neutral font-mono';
                    badgeStatus.innerText = 'Standby • ' + targetTx.id;
                }
                if (cardNarrativa) {
                    cardNarrativa.style.background = 'var(--color-primary)';
                    cardNarrativa.style.borderLeft = '4px solid var(--color-info)';
                }
                if (iconeNarrativa) {
                    iconeNarrativa.innerHTML = `<svg class="icon text-info" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>`;
                }
                if (tituloNarrativa) {
                    tituloNarrativa.innerText = `Amostra Pronta para Avaliação (${targetTx.id}):`;
                }
                if (textoNarrativa) {
                    textoNarrativa.innerHTML = `Transação carregada: <strong>${targetTx.description}</strong> (${totalFormatado}, Horário: ${horaFormatada}). Finalize o pedido no <strong>Checkout (Passo 2)</strong> para executar o ensemble de IA e inspecionar a atribuição detalhada dos pesos e correlações XAI.`;
                }
            } else {
                const prob = result.fraud_probability !== undefined ? result.fraud_probability : (targetTx.expected_category === 'fraud' ? 0.98 : 0.02);
                const probPercent = (prob * 100).toFixed(1);

                if (prob >= 0.75) {
                    // CENÁRIO 1: FRAUDE CRÍTICA
                    if (badgeStatus) {
                        badgeStatus.className = 'badge-status badge-danger font-mono';
                        badgeStatus.innerText = `BLOQUEIO CRÍTICO • SCORE ${probPercent}%`;
                    }
                    if (cardNarrativa) {
                        cardNarrativa.style.background = 'rgba(239, 68, 68, 0.08)';
                        cardNarrativa.style.borderLeft = '4px solid var(--color-destructive)';
                    }
                    if (iconeNarrativa) {
                        iconeNarrativa.innerHTML = `<svg class="icon text-danger" viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`;
                    }
                    if (tituloNarrativa) {
                        tituloNarrativa.innerText = `Racional XAI: Violação Múltipla de Limiares Críticos (Célula 41)`;
                    }
                    if (textoNarrativa) {
                        textoNarrativa.innerHTML = `O modelo identificou que os componentes latentes <code>V14</code> (${targetTx.V14.toFixed(2)}) e <code>V17</code> (${targetTx.V17.toFixed(2)}) sofreram um deslocamento negativo extremo (< -2.5), rompendo a assinatura histórica do titular. Aliado ao disparo em <code>V4</code> (+${targetTx.V4.toFixed(2)}) e ao valor/horário (${horaFormatada}), a probabilidade de fraude alcançou <strong>${probPercent}%</strong>, superando o teto de 75% e disparando o <strong>bloqueio preventivo sem tolerância</strong>.`;
                    }
                } else if (prob >= 0.50) {
                    // CENÁRIO 2: 2FA / ZONA AMARELA
                    if (badgeStatus) {
                        badgeStatus.className = 'badge-status badge-warning font-mono';
                        badgeStatus.innerText = `DESAFIO 2FA • SCORE ${probPercent}%`;
                    }
                    if (cardNarrativa) {
                        cardNarrativa.style.background = 'rgba(245, 158, 11, 0.08)';
                        cardNarrativa.style.borderLeft = '4px solid var(--color-warning)';
                    }
                    if (iconeNarrativa) {
                        iconeNarrativa.innerHTML = `<svg class="icon text-warning" viewBox="0 0 24 24"><rect width="14" height="20" x="5" y="2" rx="2"/><line x1="12" y1="18" x2="12.01" y2="18"/></svg>`;
                    }
                    if (tituloNarrativa) {
                        tituloNarrativa.innerText = `Racional XAI: Anomalia Intermediária Exige Prova de Posse (2FA)`;
                    }
                    if (textoNarrativa) {
                        textoNarrativa.innerHTML = `A transação apresentou pontuação de risco intermediária (<strong>${probPercent}%</strong>). As variáveis chave (ex: <code>V14</code> em ${targetTx.V14.toFixed(2)} e <code>V12</code> em ${targetTx.V12.toFixed(2)}) apresentaram desvios parciais em relação ao comportamento padrão, insuficientes para bloqueio automático direto, mas arriscados para aprovação imediata. O motor de regras direcionou o fluxo para a <strong>Autenticação em Duas Etapas (SMS) com temporizador ativo</strong> para evitar atrito desnecessário com clientes legítimos.`;
                    }
                } else {
                    // CENÁRIO 3: APROVADO / SEGURO
                    if (badgeStatus) {
                        badgeStatus.className = 'badge-status badge-success font-mono';
                        badgeStatus.innerText = `APROVADO • SCORE ${probPercent}%`;
                    }
                    if (cardNarrativa) {
                        cardNarrativa.style.background = 'rgba(34, 197, 94, 0.08)';
                        cardNarrativa.style.borderLeft = '4px solid var(--color-accent)';
                    }
                    if (iconeNarrativa) {
                        iconeNarrativa.innerHTML = `<svg class="icon text-success" viewBox="0 0 24 24"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`;
                    }
                    if (tituloNarrativa) {
                        tituloNarrativa.innerText = `Racional XAI: Alta Conformidade e Baixo Risco`;
                    }
                    if (textoNarrativa) {
                        textoNarrativa.innerHTML = `Todas as variáveis discriminantes permaneceram dentro das faixas estatísticas de segurança: <code>V17</code> (${targetTx.V17.toFixed(2)}) e <code>V14</code> (${targetTx.V14.toFixed(2)}) estão em conformidade com compras autênticas, e o fator protetor <code>V16</code> (+${targetTx.V16.toFixed(2)}) amorteceu o score para apenas <strong>${probPercent}%</strong>. O pagamento de ${totalFormatado} foi <strong>aprovado instantaneamente com segurança confirmada</strong>.`;
                    }
                }
            }

            // Tabela Detalhada de Atribuição de Atributos Chave (Top Fatores Cell 41)
            if (tabelaBody) {
                const featuresXai = [
                    {
                        nome: 'V17 (Assinatura do Portador)',
                        valor: targetTx.V17.toFixed(4),
                        peso: 'r = -0.3265 (Top 1)',
                        isRisk: targetTx.V17 < -1.5,
                        barWidth: targetTx.V17 < -1.5 ? Math.min(100, Math.abs(targetTx.V17) * 20) : 25,
                        barClass: targetTx.V17 < -1.5 ? 'xai-bar-danger' : (targetTx.V17 < 0 ? 'xai-bar-warning' : 'xai-bar-success'),
                        interpretacao: targetTx.V17 < -1.5 ? '⚠️ Desvio severo no padrão biométrico/comportamental' : (targetTx.V17 < 0 ? 'Oscilação leve na assinatura do portador' : '✓ Assinatura altamente compatível com histórico habitual')
                    },
                    {
                        nome: 'V14 (Integridade do Cartão)',
                        valor: targetTx.V14.toFixed(4),
                        peso: 'r = -0.3025 (Top 2)',
                        isRisk: targetTx.V14 < -2.0,
                        barWidth: targetTx.V14 < -2.0 ? Math.min(100, Math.abs(targetTx.V14) * 18) : 20,
                        barClass: targetTx.V14 < -2.0 ? 'xai-bar-danger' : (targetTx.V14 < -0.5 ? 'xai-bar-warning' : 'xai-bar-success'),
                        interpretacao: targetTx.V14 < -2.0 ? '🚨 Alerta crítico de clonagem ou dados comprometidos' : (targetTx.V14 < -0.5 ? 'Dispersão intermediária em análise' : '✓ Integridade do token de cartão validada')
                    },
                    {
                        nome: 'V12 (Consistência de Saldo & Limite)',
                        valor: targetTx.V12.toFixed(4),
                        peso: 'r = -0.2606 (Top 3)',
                        isRisk: targetTx.V12 < -1.5,
                        barWidth: targetTx.V12 < -1.5 ? Math.min(100, Math.abs(targetTx.V12) * 22) : 22,
                        barClass: targetTx.V12 < -1.5 ? 'xai-bar-danger' : 'xai-bar-success',
                        interpretacao: targetTx.V12 < -1.5 ? '⚠️ Padrão anômalo de esgotamento/débito rápido' : '✓ Histórico de saldo e limites preservado'
                    },
                    {
                        nome: 'V10 (Canal & Dispositivo Emissor)',
                        valor: targetTx.V10.toFixed(4),
                        peso: 'r = -0.2169 (Top 4)',
                        isRisk: targetTx.V10 < -1.2,
                        barWidth: targetTx.V10 < -1.2 ? Math.min(100, Math.abs(targetTx.V10) * 24) : 18,
                        barClass: targetTx.V10 < -1.2 ? 'xai-bar-danger' : 'xai-bar-success',
                        interpretacao: targetTx.V10 < -1.2 ? '⚠️ IP / fingerprint de dispositivo atípico' : '✓ Dispositivo e canal verificados'
                    },
                    {
                        nome: 'V4 (Aceleração / Frequência)',
                        valor: (targetTx.V4 >= 0 ? '+' : '') + targetTx.V4.toFixed(4),
                        peso: 'r = +0.1334 (Gatilho Positivo)',
                        isRisk: targetTx.V4 > 1.8,
                        barWidth: targetTx.V4 > 1.8 ? Math.min(100, targetTx.V4 * 25) : 15,
                        barClass: targetTx.V4 > 1.8 ? 'xai-bar-danger' : 'xai-bar-success',
                        interpretacao: targetTx.V4 > 1.8 ? '🚨 Surto de requisições sequenciais (Ataque automatizado)' : '✓ Intervalo normal entre compras'
                    },
                    {
                        nome: 'V16 (Fator Protetor de Compra Habitual)',
                        valor: (targetTx.V16 >= 0 ? '+' : '') + targetTx.V16.toFixed(4),
                        peso: 'r = +0.1965 (Amortecedor)',
                        isRisk: targetTx.V16 < -1.0,
                        barWidth: targetTx.V16 >= 0 ? Math.min(100, (targetTx.V16 + 1) * 30) : 30,
                        barClass: targetTx.V16 >= 0 ? 'xai-bar-success' : 'xai-bar-warning',
                        interpretacao: targetTx.V16 >= 0 ? '✓ Peso protetivo favorável (reduz falsos-positivos)' : '⚠️ Redução da margem protetiva habitual'
                    },
                    {
                        nome: 'Time (Janela de Horário 48h)',
                        valor: `${targetTx.Time}s (${horaFormatada})`,
                        peso: 'Janela Temporal',
                        isRisk: hora < 6,
                        barWidth: hora < 6 ? 85 : 30,
                        barClass: hora < 6 ? 'xai-bar-warning' : 'xai-bar-success',
                        interpretacao: hora < 6 ? '⚠️ Transação de madrugada (00h-05h): Pico estatístico de risco' : '✓ Horário diurno/comercial de alta segurança'
                    },
                    {
                        nome: 'Amount (Valor na Fatura)',
                        valor: totalFormatado,
                        peso: 'RobustScaler',
                        isRisk: targetTx.Amount > 1000,
                        barWidth: targetTx.Amount > 1000 ? 75 : 35,
                        barClass: targetTx.Amount > 1000 ? 'xai-bar-warning' : 'xai-bar-success',
                        interpretacao: targetTx.Amount > 1000 ? 'Ticket elevado: Exige maior rigor estatístico' : '✓ Valor compatível com a rotina do e-commerce'
                    }
                ];

                tabelaBody.innerHTML = featuresXai.map(f => `
                    <tr>
                        <td class="font-mono text-white fw-bold">${f.nome}</td>
                        <td class="font-mono ${f.isRisk ? 'text-danger fw-bold' : 'text-success'}">${f.valor}</td>
                        <td class="font-mono text-secondary small">${f.peso}</td>
                        <td>
                            <div class="xai-bar-track">
                                <div class="xai-bar-fill ${f.barClass}" style="width: ${Math.round(f.barWidth)}%;"></div>
                            </div>
                        </td>
                        <td class="small ${f.isRisk ? 'text-danger' : 'text-light'}">${f.interpretacao}</td>
                    </tr>
                `).join('');
            }
        }

        async function executarAuditoria(simulateDrift) {
            const loadEl = document.getElementById('driftLoading');
            if (loadEl) loadEl.style.display = 'block';
            const cardEl = document.getElementById('driftResultadoCard');
            if (cardEl) cardEl.style.display = 'none';

            try {
                const res = await fetch('/drift?simulate=' + simulateDrift);
                const data = await res.json();

                const alertBox = document.getElementById('alertBoxDrift');
                const title = document.getElementById('alertDriftTitle');
                const msg = document.getElementById('alertDriftMsg');
                const icon = document.getElementById('alertDriftIcon');

                if (data.action_required) {
                    if (alertBox) {
                        alertBox.style.background = "rgba(239, 68, 68, 0.15)";
                        alertBox.style.borderColor = "var(--color-destructive)";
                    }
                    if (title) {
                        title.className = "h6 mb-1 fw-bold text-danger";
                        title.innerText = "ALERTA DE DEGRAVAÇÃO POPULACIONAL (DATA DRIFT DETECTADO)";
                    }
                    if (msg) {
                        msg.className = "mb-0 small text-light";
                        msg.innerText = `${data.recommendation} — (${data.variables_in_drift} de ${data.total_variables_analyzed} variáveis com p < 0.05 / PSI >= 0.20).`;
                    }
                    if (icon) icon.innerHTML = `<svg class="icon-xl text-danger" viewBox="0 0 24 24"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`;
                } else {
                    if (alertBox) {
                        alertBox.style.background = "rgba(34, 197, 94, 0.15)";
                        alertBox.style.borderColor = "var(--color-accent)";
                    }
                    if (title) {
                        title.className = "h6 mb-1 fw-bold text-success";
                        title.innerText = "OPERAÇÃO ESTÁVEL (SEM DERIVA SIGNIFICATIVA)";
                    }
                    if (msg) {
                        msg.className = "mb-0 small text-light";
                        msg.innerText = `${data.recommendation} — Baseline histórica preservada (${data.variables_in_drift} variáveis com desvio leve).`;
                    }
                    if (icon) icon.innerHTML = `<svg class="icon-xl text-success" viewBox="0 0 24 24"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`;
                }

                const tbody = document.getElementById('tabelaDriftBody');
                if (tbody) {
                    tbody.innerHTML = "";
                    for (const [feat, metrics] of Object.entries(data.detailed_metrics)) {
                        const tr = document.createElement('tr');
                        const isDrift = metrics.drift_detected;
                        const statusBadge = isDrift 
                            ? `<span class="badge-status badge-danger"><svg class="icon" viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg> DRIFT</span>` 
                            : `<span class="badge-status badge-success"><svg class="icon" viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></svg> ESTÁVEL</span>`;

                        tr.innerHTML = `
                            <td><strong>${feat}</strong></td>
                            <td>${metrics.ks_statistic.toFixed(4)}</td>
                            <td class="${metrics.p_value < 0.05 ? 'text-danger fw-bold' : ''}">${metrics.p_value.toFixed(5)}</td>
                            <td class="${metrics.psi >= 0.20 ? 'text-danger fw-bold' : ''}">${metrics.psi.toFixed(4)}</td>
                            <td>${statusBadge}</td>
                        `;
                        tbody.appendChild(tr);
                    }
                }

                if (cardEl) cardEl.style.display = 'block';
            } catch (err) {
                alert("Erro ao rodar auditoria de drift: " + err);
            } finally {
                if (loadEl) loadEl.style.display = 'none';
            }
        }

        // Inicialização ao carregar a página
        function inicializarLoja() {
            sortearNovoCartao();
            renderizarCatalogo();
            popularSelectManual();
            atualizarCardAmostra(activeTx);
            atualizarPainelXAI(activeTx, null);
            selecionarProduto('prod-1');
            switchTab('produtos');
        }

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', inicializarLoja);
        } else {
            inicializarLoja();
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
    <style>
        body { background: #020617; }
        .swagger-ui { filter: invert(88%) hue-rotate(180deg); }
        .swagger-ui .topbar { display: none; }
    </style>
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

CHECKOUT_PATH = os.path.join(BASE_DIR, "checkout.html")
if not os.path.exists(CHECKOUT_PATH):
    CHECKOUT_PATH = os.path.join(os.path.dirname(BASE_DIR), "checkout.html")

CHECKOUT_HTML = ""
if os.path.exists(CHECKOUT_PATH):
    with open(CHECKOUT_PATH, "r", encoding="utf-8") as f:
        CHECKOUT_HTML = f.read()

SAMPLES_PATH = os.path.join(BASE_DIR, "transactions_sample.json")
if not os.path.exists(SAMPLES_PATH):
    SAMPLES_PATH = os.path.join(os.path.dirname(BASE_DIR), "transactions_sample.json")

SAMPLE_TRANSACTIONS = []
if os.path.exists(SAMPLES_PATH):
    with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
        SAMPLE_TRANSACTIONS = json.load(f)

def get_health_json(request: Request = None):
    res = {
        "status": "online",
        "service": "MegaLoja Fraud Guard API (Vercel Serverless)",
        "model_loaded": pipeline_model is not None,
        "model_name": metadata.get("model_name", "MegaLoja-Fraud-Detector-Bagging-LogisticRegression"),
        "algorithm": metadata.get("algorithm", "BaggingClassifier (base_estimator=LogisticRegression)"),
        "model_version": metadata.get("version", "1.2.0"),
        "drift_monitoring": "Ativo (Kolmogorov-Smirnov & PSI)",
        "checkout_vitrine": "/checkout",
        "sample_transactions": "/samples",
        "documentation": "/docs"
    }
    if request:
        res["debug_headers"] = {k: v for k, v in request.headers.items() if "auth" not in k.lower()}
        res["debug_url"] = str(request.url)
        res["debug_scope_path"] = request.scope.get("path")
    return res

def get_requested_path(request: Request) -> str:
    """Extrai o caminho original da requisição mesmo sob rewrite do Vercel Serverless."""
    q = request.query_params.get("__path", "")
    if q:
        return q.lower()
    for header in ["x-matched-path", "x-forwarded-uri", "x-invoke-path"]:
        val = request.headers.get(header)
        if val:
            return val.lower()
    return request.url.path.lower()

# 1. Rota raiz: serve o Dashboard HTML interativo se for navegador, ou JSON se for API
@app.get("/", response_class=HTMLResponse)
@app.get("/index", response_class=HTMLResponse)
@app.get("/api/index.py")
def root_view(request: Request):
    req_path = get_requested_path(request)
    
    if "checkout" in req_path:
        return HTMLResponse(content=CHECKOUT_HTML)
        
    if "sample" in req_path:
        return JSONResponse(content=SAMPLE_TRANSACTIONS)
    
    if "drift" in req_path:
        simulate = "simulate=true" in str(request.query_params).lower()
        if run_drift_analysis:
            return JSONResponse(content=run_drift_analysis(simulate_drift=simulate))
        return JSONResponse(content={"status": "Drift monitor indisponível"})
        
    if "doc" in req_path:
        return HTMLResponse(content=SWAGGER_HTML)
        
    if "openapi" in req_path:
        return app.openapi()
        
    if "health" in req_path:
        return JSONResponse(content=get_health_json(request))

    accept = request.headers.get("accept", "")
    if "application/json" in accept and "text/html" not in accept:
        return JSONResponse(content=get_health_json(request))
        
    return HTMLResponse(content=HTML_DASHBOARD)

# 2. Endpoint da Vitrine de Checkout React
@app.get("/checkout", response_class=HTMLResponse)
@app.get("/api/checkout", response_class=HTMLResponse)
@app.get("/api/index.py/checkout", response_class=HTMLResponse)
def checkout_view():
    return HTMLResponse(content=CHECKOUT_HTML)

# 3. Endpoint de Amostras de Transações (15 amostras: 10 normais, 5 fraudes)
@app.get("/samples")
@app.get("/api/samples")
@app.get("/api/index.py/samples")
def samples_view():
    return JSONResponse(content=SAMPLE_TRANSACTIONS)

# 4. Endpoint de Health Check
@app.get("/health")
@app.get("/api/health")
def health_endpoint(request: Request):
    return JSONResponse(content=get_health_json(request))

# 5. Documentação Swagger
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

# 6. Endpoint de Auditoria de Drift e Degradação (Etapa 5 - Slide 18)
@app.get("/drift")
@app.get("/api/drift")
def drift_endpoint(simulate: bool = False):
    if run_drift_analysis:
        return JSONResponse(content=run_drift_analysis(simulate_drift=simulate))
    return JSONResponse(content={"status": "Drift monitor indisponível"})

# 7. Endpoint de Inferência Preditiva (POST)
@app.post("/predict", response_model=PredictionResponse)
@app.post("/api/predict", response_model=PredictionResponse)
@app.post("/api/index.py/predict", response_model=PredictionResponse)
def predict_endpoint(transaction: TransactionInput):
    res = run_inference(transaction.model_dump())
    return PredictionResponse(**res)

# 8. Roteador Universal Catch-All
@app.api_route("/{full_path:path}", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def universal_router(request: Request, full_path: str):
    clean_path = full_path.lower().strip("/")
    req_path = get_requested_path(request)
    combined = f"{clean_path} {req_path}"
    
    if request.method == "POST":
        try:
            body = await request.json()
            return PredictionResponse(**run_inference(body))
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
            
    if "checkout" in combined:
        return HTMLResponse(content=CHECKOUT_HTML)
        
    if "sample" in combined:
        return JSONResponse(content=SAMPLE_TRANSACTIONS)

    if "drift" in combined:
        simulate = "simulate=true" in str(request.query_params).lower()
        if run_drift_analysis:
            return JSONResponse(content=run_drift_analysis(simulate_drift=simulate))
        return JSONResponse(content={"status": "Drift monitor indisponível"})
        
    if "doc" in combined:
        return HTMLResponse(content=SWAGGER_HTML)
        
    if "openapi" in combined:
        return app.openapi()
        
    if "health" in combined:
        return JSONResponse(content=get_health_json(request))
        
    accept = request.headers.get("accept", "")
    if "application/json" in accept and "text/html" not in accept:
        return JSONResponse(content=get_health_json(request))
        
    return HTMLResponse(content=HTML_DASHBOARD)
