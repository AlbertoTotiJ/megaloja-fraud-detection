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
                                <span class="font-mono small text-secondary">CARTÃO VIRTUAL</span>
                            </div>
                            <div class="font-mono fs-5 fw-bold mb-2 text-white" style="letter-spacing: 2px;">
                                •••• •••• •••• 4242
                            </div>
                            <div class="d-flex justify-content-between align-items-end font-mono small">
                                <div>
                                    <span class="text-secondary" style="font-size: 0.65rem; display: block;">TITULAR</span>
                                    <span class="text-white fw-bold">ALBERTO T FERRARI</span>
                                </div>
                                <div>
                                    <span class="text-secondary" style="font-size: 0.65rem; display: block;">VALIDADE</span>
                                    <span class="text-white">11/29</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Coluna 2: Injeção de Risco Antifraude & Finalizar Compra -->
                <div class="col-lg-6">
                    <div class="card-custom p-4 mb-4">
                        <div class="d-flex justify-content-between align-items-center mb-3 pb-2 border-bottom" style="border-color: var(--color-border) !important;">
                            <div>
                                <h3 class="h6 mb-0 text-white fw-bold">Perfil de Risco da Transação</h3>
                                <span class="text-secondary small">Selecione o comportamento para testar a reação</span>
                            </div>
                            <span class="badge-status badge-info">FastAPI ML</span>
                        </div>

                        <!-- Cenários Pré-configurados -->
                        <div class="d-flex flex-column gap-2 mb-3">
                            <button type="button" class="preset-chip active" id="chip-normal" onclick="carregarPreset('normal')">
                                <div class="d-flex align-items-center gap-2 mb-1">
                                    <svg class="icon text-success" viewBox="0 0 24 24"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                                    <strong class="text-white small">🟢 Compra Habitual Segura (Score &lt; 50%)</strong>
                                </div>
                                <div class="text-secondary small font-mono">Cliente habitual • Horário comercial • Cartão verificado</div>
                            </button>

                            <button type="button" class="preset-chip" id="chip-suspeita" onclick="carregarPreset('suspeita')">
                                <div class="d-flex align-items-center gap-2 mb-1">
                                    <svg class="icon text-warning" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                                    <strong class="text-white small">🟡 Zona Amarela / Risco Médio (Desafio 2FA SMS)</strong>
                                </div>
                                <div class="text-secondary small font-mono">Score 50% a 74% • Compra noturna • Requer SMS com temporizador</div>
                            </button>

                            <button type="button" class="preset-chip" id="chip-fraude" onclick="carregarPreset('fraude')">
                                <div class="d-flex align-items-center gap-2 mb-1">
                                    <svg class="icon text-danger" viewBox="0 0 24 24"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                                    <strong class="text-white small">🔴 Tentativa de Fraude Crítica (Score &ge; 75%)</strong>
                                </div>
                                <div class="text-secondary small font-mono">Padrões Cell 41 • Madrugada • Bloqueio imediato por segurança</div>
                            </button>
                        </div>

                        <!-- Parâmetros brutos -->
                        <div class="row g-2 mb-3">
                            <div class="col-6">
                                <label for="time_val" class="form-label font-mono">Tempo (Segundos):</label>
                                <input type="number" id="time_val" class="form-control-custom" value="45000.0" step="1" oninput="atualizarHoraEstimada()">
                                <div class="text-secondary small mt-1 font-mono" id="time_hora_display" style="font-size: 0.72rem;">12:30:00 (Hora 12)</div>
                            </div>
                            <div class="col-6">
                                <label for="amount_val" class="form-label font-mono">Valor Amostra na Fatura (R$):</label>
                                <input type="number" id="amount_val" class="form-control-custom" value="89.90" step="0.01">
                                <div class="text-secondary small mt-1" style="font-size: 0.72rem;">Amostra Calibrada (RobustScaler)</div>
                            </div>
                        </div>

                        <!-- Sanfona PCA V1..V28 -->
                        <div class="mb-3">
                            <button type="button" class="btn-secondary-custom btn-sm w-100 justify-content-between" onclick="togglePCA()" id="btnTogglePCA">
                                <span class="d-flex align-items-center gap-2">
                                    <svg class="icon" viewBox="0 0 24 24"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>
                                    Componentes PCA (V1 a V28)
                                </span>
                                <span id="pcaStatusText" class="text-secondary small">Expandir &darr;</span>
                            </button>
                            <div id="pcaGridContainer" class="mt-2 p-2 card-custom" style="display: none; background: #070b16;">
                                <div class="d-flex justify-content-between align-items-center mb-1">
                                    <span class="small text-secondary font-mono" style="font-size: 0.72rem;">28 Variáveis Criptografadas</span>
                                    <button type="button" class="btn btn-outline-secondary btn-sm py-0 font-mono" style="font-size: 0.7rem;" onclick="zerarPCAs()">Zerar V1..V28</button>
                                </div>
                                <div class="row g-1" id="pcaInputsContainer"></div>
                            </div>
                        </div>

                        <!-- Botão CTA Principal de Finalizar Compra -->
                        <button id="btnFinalizarCompra" class="btn-cta w-100 py-3" onclick="processarCheckout()">
                            <svg class="icon-lg" viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/></svg>
                            <span>Finalizar Compra e Processar Pagamento &rarr;</span>
                        </button>
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
                        <span class="badge-status badge-info">CRISP-DM MLOps</span>
                        <h3 class="h6 mb-0 text-white fw-bold">Auditoria do Pipeline, Modelagem & Monitoramento de Data Drift</h3>
                    </div>
                    <span id="crispSecaoStatus" class="text-secondary small font-mono">Expandir Detalhes &darr;</span>
                </div>

                <div id="crispSecaoBody" class="mt-3 pt-3 border-top" style="display: none; border-color: var(--color-border) !important;">
                    <p class="text-secondary small mb-3">
                        Implementação das etapas metodológicas do Projeto Integrador V ministrado pelo <strong>Prof. Dr. Sidney Carlos Ferrari</strong> (FATEC Ourinhos).
                    </p>

                    <div class="row g-3 mb-4">
                        <div class="col-md-4">
                            <div class="card-custom p-3 h-100" style="background: var(--color-muted);">
                                <span class="badge-status badge-info mb-1">Etapas 1 & 2</span>
                                <h4 class="h6 text-white fw-bold">Negócio & Dados</h4>
                                <p class="text-secondary small mb-0">284.807 transações da MegaLoja balanceadas via Under-sampling supervisionado (Cell 41).</p>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card-custom p-3 h-100" style="background: var(--color-muted);">
                                <span class="badge-status badge-success mb-1">Etapas 3 & 4</span>
                                <h4 class="h6 text-white fw-bold">Modelagem & Deploy</h4>
                                <p class="text-secondary small mb-0">BaggingClassifier (10 Regressões Logísticas), modelo serializado com latência &lt; 15ms.</p>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card-custom p-3 h-100" style="background: var(--color-muted);">
                                <span class="badge-status badge-warning mb-1">Etapa 5 (Slide 18)</span>
                                <h4 class="h6 text-white fw-bold">Drift & MLOps</h4>
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

        // Amostras Calibradas do Modelo Treinado (Prof. Dr. Sidney Ferrari)
        // 1. Fraude Crítica Real (Amostra Cell 41 do notebook) -> Score ~98.3% (Bloqueio Imediato)
        const fraudValues = {
            Time: 406.0, Amount: 149.62,
            V1: -2.31, V2: 1.95, V3: -1.6, V4: 3.99, V5: -0.52, V6: -1.42, V7: -2.53, V8: 1.39,
            V9: -2.77, V10: -2.77, V11: 3.2, V12: -2.89, V13: -0.59, V14: -4.28, V15: 0.38,
            V16: -1.14, V17: -2.83, V18: -0.01, V19: 0.41, V20: 0.12, V21: 0.51, V22: -0.03,
            V23: -0.46, V24: 0.32, V25: 0.04, V26: 0.17, V27: 0.26, V28: -0.14
        };

        // 2. Compra Cotidiana Segura (Transação legítima de alta confiança) -> Score ~0.9% (Pagamento Aprovado)
        const normalValues = {
            Time: 45000.0, Amount: 89.90,
            V1: 0.0, V2: 0.0, V3: 0.0, V4: 0.0, V5: -0.6, V6: 0.0, V7: 1.4, V8: -0.7,
            V9: 0.0, V10: 0.0, V11: 0.0, V12: 1.3, V13: 0.0, V14: 0.0, V15: -0.7,
            V16: 1.8, V17: 0.0, V18: 0.0, V19: 0.0, V20: 0.0, V21: -0.6, V22: 0.0,
            V23: 1.0, V24: 0.0, V25: 1.6, V26: 1.2, V27: 0.0, V28: 0.0
        };

        // 3. Zona Amarela / Risco Moderado (Transação atípica em horário noturno) -> Score ~52.1% (2FA SMS)
        const suspeitaValues = {
            Time: 65000.0, Amount: 380.00,
            V1: 0.0, V2: 0.0, V3: 0.0, V4: 0.0, V5: -0.05, V6: 0.0, V7: 0.45, V8: -0.1,
            V9: 0.0, V10: 0.0, V11: 0.0, V12: 0.45, V13: 0.0, V14: 0.0, V15: -0.1,
            V16: 0.78, V17: 0.0, V18: 0.0, V19: 0.0, V20: 0.0, V21: -0.05, V22: 0.0,
            V23: 0.3, V24: 0.0, V25: 0.65, V26: 0.4, V27: 0.0, V28: 0.0
        };

        let currentPCA = { ...normalValues };

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
                btn.innerHTML = `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Avaliando Risco Antifraude...`;
            }

            currentPCA.Time = parseFloat(document.getElementById('time_val').value) || 50000.0;
            currentPCA.Amount = parseFloat(document.getElementById('amount_val').value) || (produtoAtual.preco - 100.0);

            for (let i = 1; i <= 28; i++) {
                const el = document.getElementById('pca_v' + i);
                if (el) currentPCA['V' + i] = parseFloat(el.value) || 0.0;
            }

            const t0 = performance.now();
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(currentPCA)
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
                    btn.innerHTML = `<svg class="icon-lg" viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/></svg> <span>Finalizar Compra e Processar Pagamento &rarr;</span>`;
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
                            O motor antifraude identificou desvios comportamentais severos compatíveis com clonagem e ataque automatizado de cartões. A compra de <strong>${produtoAtual.nome}</strong> no valor de <strong>${totalFormatado}</strong> foi barrada em milissegundos.
                        </p>

                        <!-- Tabela de Auditoria do Bloqueio -->
                        <div class="p-3 rounded mb-4 text-start font-mono small mx-auto" style="max-width: 650px; background: var(--color-primary); border: 1px solid var(--color-border);">
                            <div class="d-flex justify-content-between mb-1">
                                <span class="text-secondary">Parecer do Modelo:</span>
                                <strong class="text-danger">FRAUDE DETECTADA</strong>
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
                        </div>
                    </div>
                `;
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
                            Identificamos um padrão de compra atípico para o produto <strong>${produtoAtual.nome}</strong> (${totalFormatado}). Enviamos um código de 6 dígitos para o celular cadastrado com final <strong>(11) 98***-**42</strong>.
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

                                <div class="d-flex gap-2">
                                    <button type="button" class="btn-secondary-custom flex-grow-1" onclick="switchTab('checkout')">
                                        Voltar ao Checkout
                                    </button>
                                    <button type="submit" class="btn-cta flex-grow-1" style="background: var(--color-warning); color: #000;">
                                        Confirmar Código SMS
                                    </button>
                                </div>
                            </form>
                        </div>
                    </div>
                `;

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
                        Transação autorizada com sucesso pelo emissor do cartão. O pedido do produto <strong>${produtoAtual.nome}</strong> no valor de <strong>${totalFormatado}</strong> foi confirmado e liberado para expedição imediata!
                    </p>

                    <!-- Tabela de Auditoria do Pedido Aprovado -->
                    <div class="p-3 rounded mb-4 text-start font-mono small mx-auto" style="max-width: 650px; background: var(--color-primary); border: 1px solid var(--color-border);">
                        <div class="d-flex justify-content-between mb-1">
                            <span class="text-secondary">Status da Transação:</span>
                            <strong class="text-success">APROVADA (NÃO É FRAUDE)</strong>
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
                    </div>
                </div>
            `;
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
                            Identidade do portador verificada com sucesso via token SMS 2FA (código <strong>849204</strong> validado). O pedido de <strong>${produtoAtual.nome}</strong> no valor de <strong>${totalFormatado}</strong> foi liberado para envio imediato!
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
                        </div>
                    </div>
                `;
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
                        ${motivo} A compra do produto <strong>${produtoAtual.nome}</strong> (${totalFormatado}) foi recusada e o cartão temporariamente suspenso para proteção do titular.
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
                    </div>
                </div>
            `;
        }

        // Renderiza inputs das variáveis PCA
        function renderizarPCAInputs() {
            const container = document.getElementById('pcaInputsContainer');
            if (!container) return;
            container.innerHTML = '';
            for (let i = 1; i <= 28; i++) {
                const col = document.createElement('div');
                col.className = 'col-6 col-sm-4 col-md-3 col-lg-2';
                col.innerHTML = `
                    <label for="pca_v${i}" class="form-label font-mono" style="font-size: 0.7rem;">V${i}:</label>
                    <input type="number" id="pca_v${i}" class="form-control-custom py-1 px-2" style="font-size: 0.75rem;" 
                           value="${currentPCA['V' + i] || 0.0}" step="0.01" oninput="atualizarPCAVal(${i}, this.value)">
                `;
                container.appendChild(col);
            }
        }

        function atualizarPCAVal(index, val) {
            currentPCA['V' + index] = parseFloat(val) || 0.0;
        }

        function zerarPCAs() {
            for (let i = 1; i <= 28; i++) {
                currentPCA['V' + i] = 0.0;
                const el = document.getElementById('pca_v' + i);
                if (el) el.value = "0.0";
            }
        }

        function atualizarHoraEstimada() {
            const timeEl = document.getElementById('time_val');
            const t = parseFloat(timeEl ? timeEl.value : 0) || 0;
            const hour = Math.floor(t / 3600) % 24;
            const min = Math.floor((t % 3600) / 60);
            const sec = Math.floor(t % 60);
            const pad = (n) => String(n).padStart(2, '0');
            const el = document.getElementById('time_hora_display');
            if (el) el.innerText = `Horário: ${pad(hour)}:${pad(min)}:${pad(sec)} (Hora ${hour})`;
        }

        function carregarPreset(tipo) {
            document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));

            if (tipo === 'fraude') {
                const chip = document.getElementById('chip-fraude');
                if (chip) chip.classList.add('active');
                currentPCA = { ...fraudValues };
            } else if (tipo === 'normal') {
                const chip = document.getElementById('chip-normal');
                if (chip) chip.classList.add('active');
                currentPCA = { ...normalValues };
            } else if (tipo === 'suspeita') {
                const chip = document.getElementById('chip-suspeita');
                if (chip) chip.classList.add('active');
                currentPCA = { ...suspeitaValues };
            }

            const timeEl = document.getElementById('time_val');
            if (timeEl) timeEl.value = currentPCA.Time;
            const amountEl = document.getElementById('amount_val');
            if (amountEl) amountEl.value = currentPCA.Amount.toFixed(2);
            atualizarHoraEstimada();

            for (let i = 1; i <= 28; i++) {
                const el = document.getElementById('pca_v' + i);
                if (el) el.value = currentPCA['V' + i] !== undefined ? currentPCA['V' + i] : 0.0;
            }
        }

        function togglePCA() {
            const grid = document.getElementById('pcaGridContainer');
            const status = document.getElementById('pcaStatusText');
            if (!grid || !status) return;
            if (grid.style.display === 'none') {
                grid.style.display = 'block';
                status.innerHTML = 'Recolher &uarr;';
            } else {
                grid.style.display = 'none';
                status.innerHTML = 'Expandir &darr;';
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
            renderizarCatalogo();
            renderizarPCAInputs();
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
