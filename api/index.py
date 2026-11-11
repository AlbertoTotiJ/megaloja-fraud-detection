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
    <title>MegaLoja Fraud Guard — Sistema Preditivo de Fraudes</title>
    <!-- Google Fonts: Fira Code + Fira Sans (Design System Master) -->
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
            --shadow-glow-accent: 0 0 15px rgba(34, 197, 94, 0.25);
            --shadow-glow-danger: 0 0 15px rgba(239, 68, 68, 0.35);
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
            padding: 2.2rem 1rem 1.8rem;
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

        /* Preset Cards */
        .preset-chip {
            background: var(--color-muted);
            border: 1px solid var(--color-border);
            border-radius: 8px;
            padding: 10px 14px;
            cursor: pointer;
            transition: all 200ms ease;
            text-align: left;
            flex: 1;
            min-width: 180px;
        }
        .preset-chip:hover {
            border-color: var(--color-info);
            background: #22293f;
            transform: translateY(-2px);
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
            padding: 14px 20px;
            cursor: pointer;
            transition: all 200ms ease;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            font-family: 'Fira Sans', sans-serif;
            font-size: 1.05rem;
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
            padding: 10px 16px;
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
            width: 42px;
            height: 42px;
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
            padding: 12px 14px;
            border-bottom: 1px solid var(--color-border);
            text-align: left;
        }
        .table-custom td {
            padding: 12px 14px;
            border-bottom: 1px solid #1e293b;
            font-family: 'Fira Code', monospace;
        }
        .table-custom tr:hover td {
            background: rgba(30, 41, 59, 0.5);
        }

        /* Gauge progress bar */
        .gauge-bar-wrapper {
            background: var(--color-primary);
            height: 14px;
            border-radius: 7px;
            overflow: hidden;
            border: 1px solid var(--color-border);
            margin: 12px 0;
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

        /* Reduced motion */
        @media (prefers-reduced-motion: reduce) {
            * {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
        }
    </style>
</head>
<body>

    <!-- Header Corporativo Enterprise Gateway -->
    <header class="header-glow">
        <div class="container" style="max-width: 1000px;">
            <div class="d-flex flex-wrap justify-content-between align-items-center gap-3">
                <div class="d-flex align-items-center gap-3">
                    <div style="background: rgba(34, 197, 94, 0.15); border: 1px solid rgba(34, 197, 94, 0.3); border-radius: 12px; padding: 12px; color: var(--color-accent);">
                        <!-- Shield Icon -->
                        <svg class="icon-xl" viewBox="0 0 24 24" aria-hidden="true">
                            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                            <path d="m9 12 2 2 4-4"/>
                        </svg>
                    </div>
                    <div>
                        <div class="d-flex align-items-center gap-2 mb-1">
                            <h1 class="h3 mb-0 fw-bold text-white">MegaLoja Fraud Guard</h1>
                            <span class="badge-status badge-success">
                                <span class="pulse-dot online"></span> v1.2.0 LIVE
                            </span>
                        </div>
                        <p class="text-secondary mb-0 small">
                            Motor Preditivo em Tempo Real & Monitoramento MLOps de Data Drift — Metodologia CRISP-DM
                        </p>
                    </div>
                </div>

                <div class="d-flex flex-column align-items-end text-end">
                    <div class="d-flex gap-2 mb-1">
                        <span class="badge-status badge-info">
                            <!-- Cloud Icon -->
                            <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/></svg>
                            Vercel Serverless
                        </span>
                        <span class="badge-status badge-neutral">
                            <!-- Cpu Icon -->
                            <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/></svg>
                            Bagging (RL x10)
                        </span>
                    </div>
                    <span class="text-secondary small font-mono">FATEC Ourinhos • Prof. Dr. Sidney C. Ferrari</span>
                </div>
            </div>
        </div>
    </header>

    <!-- Main Container -->
    <main class="container py-4" style="max-width: 1000px;">

        <!-- Navigation Tabs -->
        <nav class="nav-tabs-custom" role="tablist" aria-label="Seções do Sistema">
            <button class="tab-btn active" id="tab-btn-simulador" onclick="switchTab('simulador')" role="tab" aria-selected="true" aria-controls="tab-simulador">
                <!-- Zap Icon -->
                <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
                1. Inferência em Tempo Real (Etapas 1-4)
            </button>
            <button class="tab-btn" id="tab-btn-drift" onclick="switchTab('drift')" role="tab" aria-selected="false" aria-controls="tab-drift">
                <!-- Activity Icon -->
                <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
                2. Monitoramento de Drift & MLOps (Etapa 5)
            </button>
            <button class="tab-btn" id="tab-btn-crisp" onclick="switchTab('crisp')" role="tab" aria-selected="false" aria-controls="tab-crisp">
                <!-- Layers Icon -->
                <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
                3. Arquitetura & Governança CRISP-DM
            </button>
        </nav>

        <!-- ========================================== -->
        <!-- TAB 1: INFERÊNCIA EM TEMPO REAL -->
        <!-- ========================================== -->
        <section id="tab-simulador" role="tabpanel" aria-labelledby="tab-btn-simulador">
            <div class="card-custom p-4 mb-4">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <div>
                        <h2 class="h5 mb-1 text-white fw-bold">Simulação de Transação Bancária</h2>
                        <p class="text-secondary small mb-0">Insira os parâmetros brutos ou clique nos cenários de teste pré-configurados.</p>
                    </div>
                    <span class="badge-status badge-info">Latência Estimada: &lt; 15ms</span>
                </div>

                <!-- Presets -->
                <div class="d-flex flex-wrap gap-2 mb-4">
                    <button type="button" class="preset-chip active" id="chip-fraude" onclick="carregarPreset('fraude')">
                        <div class="d-flex align-items-center gap-2 mb-1">
                            <!-- Alert Triangle Icon -->
                            <svg class="icon text-danger" viewBox="0 0 24 24" aria-hidden="true"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                            <strong class="text-white small">Fraude Crítica (Cell 41)</strong>
                        </div>
                        <div class="text-secondary small font-mono">R$ 149,62 • Madrugada</div>
                    </button>

                    <button type="button" class="preset-chip" id="chip-normal" onclick="carregarPreset('normal')">
                        <div class="d-flex align-items-center gap-2 mb-1">
                            <!-- Check Circle Icon -->
                            <svg class="icon text-success" viewBox="0 0 24 24" aria-hidden="true"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                            <strong class="text-white small">Compra Cotidiana Segura</strong>
                        </div>
                        <div class="text-secondary small font-mono">R$ 45,00 • Tarde Comum</div>
                    </button>

                    <button type="button" class="preset-chip" id="chip-suspeita" onclick="carregarPreset('suspeita')">
                        <div class="d-flex align-items-center gap-2 mb-1">
                            <!-- Help Circle Icon -->
                            <svg class="icon text-warning" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                            <strong class="text-white small">Desafio de Segurança (2FA)</strong>
                        </div>
                        <div class="text-secondary small font-mono">R$ 1.850,00 • Noturno</div>
                    </button>
                </div>

                <!-- Form Inputs -->
                <div class="row g-3">
                    <div class="col-md-6">
                        <label for="time_val" class="form-label">Tempo Transcorrido (Segundos acumulados):</label>
                        <input type="number" id="time_val" class="form-control-custom" value="406.0" step="1" oninput="atualizarHoraEstimada()">
                        <div class="text-secondary small mt-1 font-mono" id="time_hora_display">Horário estimado: 00:06:46 (Hora 0)</div>
                    </div>
                    <div class="col-md-6">
                        <label for="amount_val" class="form-label">Valor Monetário da Transação (R$):</label>
                        <input type="number" id="amount_val" class="form-control-custom" value="149.62" step="0.01">
                        <div class="text-secondary small mt-1">Normalizado via <code>RobustScaler</code> na inferência</div>
                    </div>
                </div>

                <!-- PCA Accordion Toggle -->
                <div class="mt-3">
                    <button type="button" class="btn-secondary-custom btn-sm w-100 justify-content-between" onclick="togglePCA()" id="btnTogglePCA">
                        <span class="d-flex align-items-center gap-2">
                            <!-- Sliders Icon -->
                            <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>
                            Visualizar e Editar Componentes Criptografados PCA (V1 a V28)
                        </span>
                        <span id="pcaStatusText" class="text-secondary small">Expandir &darr;</span>
                    </button>
                </div>

                <!-- Grid de Variáveis PCA -->
                <div id="pcaGridContainer" class="mt-3 p-3 card-custom" style="display: none; background: #070b16;">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <span class="small text-secondary font-mono">28 Componentes Principais (PCA) do Cartão</span>
                        <button type="button" class="btn btn-outline-secondary btn-sm py-0 font-mono" style="font-size: 0.75rem;" onclick="zerarPCAs()">Resetar V1..V28 para 0.0</button>
                    </div>
                    <div class="row g-2" id="pcaInputsContainer"></div>
                </div>

                <!-- Submit Button -->
                <div class="mt-4">
                    <button id="btnAvaliar" class="btn-cta w-100" onclick="avaliarTransacao()">
                        <!-- Shield Check Icon -->
                        <svg class="icon-lg" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>
                        <span>Avaliar Transação em Tempo Real</span>
                    </button>
                </div>
            </div>

            <!-- RESULT CARD -->
            <div id="resultadoCard" class="card-custom p-4 mb-4" style="display: none;">
                <div class="d-flex justify-content-between align-items-center border-bottom pb-3 mb-3" style="border-color: var(--color-border) !important;">
                    <div>
                        <h3 class="h5 mb-0 text-white fw-bold">Resultado da Análise Preditiva</h3>
                        <span class="text-secondary small font-mono">Parecer emitido pelo Motor de Risco MegaLoja</span>
                    </div>
                    <div class="d-flex gap-2">
                        <span class="badge-status badge-neutral" id="metaLatencia">Latência: -- ms</span>
                        <span class="badge-status badge-info" id="metaStatusHttp">HTTP 200 OK</span>
                    </div>
                </div>

                <div class="row align-items-center g-4 py-2">
                    <div class="col-md-5 text-center border-end" style="border-color: var(--color-border) !important;">
                        <span class="text-secondary small text-uppercase font-mono">Probabilidade de Fraude</span>
                        <div id="probDisplay" class="metric-val display-4 fw-bold my-1 text-white">--%</div>
                        
                        <div class="gauge-bar-wrapper">
                            <div id="gaugeBar" class="gauge-bar-fill" style="width: 0%; background-color: var(--color-accent);"></div>
                        </div>

                        <div class="d-flex justify-content-between text-secondary small font-mono px-1">
                            <span>0% Seguro</span>
                            <span>Threshold: 50%</span>
                            <span>100% Crítico</span>
                        </div>
                    </div>

                    <div class="col-md-7">
                        <span class="text-secondary small text-uppercase font-mono">Recomendação do Sistema</span>
                        <div id="decisaoBanner" class="p-3 mt-2 rounded-3 border" style="background: var(--color-muted);">
                            <div class="d-flex align-items-start gap-3">
                                <div id="decisIncone" class="mt-1"></div>
                                <div>
                                    <h4 class="h6 mb-1 fw-bold" id="decisaoTitulo">--</h4>
                                    <p class="mb-0 small text-secondary" id="decisaoDesc">--</p>
                                </div>
                            </div>
                        </div>

                        <div class="mt-3 pt-2 d-flex flex-wrap gap-2 text-secondary small font-mono">
                            <div><strong>Modelo:</strong> <span id="resModeloNome" class="text-white">BaggingClassifier (RL)</span></div>
                            <div class="ms-auto"><strong>Ação da API:</strong> <span id="resAcao" class="text-white">Concluída</span></div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- ========================================== -->
        <!-- TAB 2: MONITORAMENTO DE DRIFT & MLOPS -->
        <!-- ========================================== -->
        <section id="tab-drift" role="tabpanel" aria-labelledby="tab-btn-drift" style="display: none;">
            <div class="card-custom p-4 mb-4">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <div>
                        <h2 class="h5 mb-1 text-white fw-bold">Auditoria Contínua de Data Drift (Etapa 5 CRISP-DM)</h2>
                        <p class="text-secondary small mb-0">Implementação em produção dos conceitos de MLOps do <strong>Slide 18</strong> (Prof. Dr. Sidney Ferrari).</p>
                    </div>
                    <span class="badge-status badge-info">
                        <!-- Eye Icon -->
                        <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>
                        Agente Sentinela
                    </span>
                </div>

                <div class="p-3 mb-4 rounded-3 border" style="background: var(--color-muted); border-color: var(--color-border) !important;">
                    <p class="text-secondary small mb-1">
                        Modelos em produção sofrem degradação por <strong>Data Drift</strong> (mudanças na distribuição dos atributos, ex.: inflação ou sazonalidade) e <strong>Concept Drift</strong> (novas estratégias de fraude).
                    </p>
                    <p class="text-secondary small mb-0 font-mono">
                        Critérios formais: <strong>Teste Kolmogorov-Smirnov</strong> (deriva se p-value &lt; 0.05) e <strong>Population Stability Index (PSI)</strong> (deriva se PSI &ge; 0.20). Se &ge; 30% das features sofrerem desvio, aciona-se o gatilho de retreinamento.
                    </p>
                </div>

                <!-- Botões de Ação de Auditoria -->
                <div class="d-flex flex-wrap gap-3 mb-4">
                    <button type="button" class="btn-secondary-custom flex-grow-1 py-3" onclick="executarAuditoria(false)" id="btnAuditarNormal">
                        <!-- Check Circle Icon -->
                        <svg class="icon text-success" viewBox="0 0 24 24" aria-hidden="true"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                        <span>Auditar Lote de Operação Normal (Baseline)</span>
                    </button>
                    <button type="button" class="btn-cta flex-grow-1 py-3" style="background: var(--color-warning); color: #000;" onclick="executarAuditoria(true)" id="btnAuditarAnomalia">
                        <!-- Alert Triangle Icon -->
                        <svg class="icon text-dark" viewBox="0 0 24 24" aria-hidden="true"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                        <span>Simular Anomalia Crítica / Black Friday (Drift)</span>
                    </button>
                </div>

                <!-- Spinner Loading -->
                <div id="driftLoading" class="text-center py-5" style="display: none;">
                    <div class="spinner-border text-success mb-3" role="status" style="width: 2.5rem; height: 2.5rem;"></div>
                    <div class="text-white font-mono small">Executando Kolmogorov-Smirnov bicaudal (scipy.stats.ks_2samp) e buckets de PSI...</div>
                    <div class="text-secondary small mt-1">Comparando distribuição histórica contra lote produtivo</div>
                </div>

                <!-- Tabela de Resultados Drift -->
                <div id="driftResultadoCard" style="display: none;">
                    <div id="alertBoxDrift" class="p-3 mb-4 rounded-3 border">
                        <div class="d-flex align-items-center gap-3">
                            <div id="alertDriftIcon"></div>
                            <div>
                                <h4 class="h6 mb-1 fw-bold" id="alertDriftTitle">--</h4>
                                <p class="mb-0 small" id="alertDriftMsg">--</p>
                            </div>
                        </div>
                    </div>

                    <div class="table-responsive card-custom">
                        <table class="table-custom">
                            <thead>
                                <tr>
                                    <th>Variável Crítica</th>
                                    <th>KS Statistic</th>
                                    <th>p-value (KS)</th>
                                    <th>PSI Score</th>
                                    <th>Status Individual</th>
                                </tr>
                            </thead>
                            <tbody id="tabelaDriftBody"></tbody>
                        </table>
                    </div>

                    <div class="d-flex justify-content-between align-items-center text-secondary small font-mono mt-3 px-1">
                        <span>Auditoria gravada em <code>drift_audit_report.json</code></span>
                        <span>scipy.stats • pandas • numpy</span>
                    </div>
                </div>
            </div>
        </section>

        <!-- ========================================== -->
        <!-- TAB 3: ARQUITETURA CRISP-DM -->
        <!-- ========================================== -->
        <section id="tab-crisp" role="tabpanel" aria-labelledby="tab-btn-crisp" style="display: none;">
            <div class="card-custom p-4 mb-4">
                <h2 class="h5 mb-2 text-white fw-bold">Ciclo Metodológico Completo (CRISP-DM)</h2>
                <p class="text-secondary small mb-4">
                    Alinhamento formal com as diretrizes do Projeto Integrador V ministrado pelo Prof. Dr. Sidney Carlos Ferrari (FATEC Ourinhos).
                </p>

                <div class="row g-3">
                    <div class="col-md-4">
                        <div class="card-custom p-3 h-100" style="background: var(--color-muted);">
                            <span class="badge-status badge-info mb-2">Etapa 1 & 2</span>
                            <h3 class="h6 text-white fw-bold">Negócio & Dados</h3>
                            <p class="text-secondary small mb-0">
                                Diagnóstico de 284.807 transações da MegaLoja. Desbalanceamento extremo corrigido via Under-sampling supervisionado (Cell 41 do notebook de referência).
                            </p>
                        </div>
                    </div>

                    <div class="col-md-4">
                        <div class="card-custom p-3 h-100" style="background: var(--color-muted);">
                            <span class="badge-status badge-success mb-2">Etapa 3 & 4</span>
                            <h3 class="h6 text-white fw-bold">Modelagem & Deploy</h3>
                            <p class="text-secondary small mb-0">
                                BaggingClassifier com 10 estimadores de Regressão Logística. Serialização ultra-leve (5 KB) compatível com Vercel Serverless (&lt; 15ms de tempo de resposta).
                            </p>
                        </div>
                    </div>

                    <div class="col-md-4">
                        <div class="card-custom p-3 h-100" style="background: var(--color-muted);">
                            <span class="badge-status badge-warning mb-2">Etapa 5 (Slide 18)</span>
                            <h3 class="h6 text-white fw-bold">Monitoramento & MLOps</h3>
                            <p class="text-secondary small mb-0">
                                Sentinela estatístico autônomo com teste KS e cálculo de PSI. Alertas automáticos para disparo de esteira de retreinamento (CI/CD).
                            </p>
                        </div>
                    </div>
                </div>

                <div class="mt-4 pt-3 border-top" style="border-color: var(--color-border) !important;">
                    <div class="d-flex flex-wrap justify-content-between align-items-center gap-2">
                        <span class="text-secondary small">Equipe de Desenvolvimento (Grupo 8): <strong>Alberto, Geovane, Gustavo</strong></span>
                        <a href="https://github.com/AlbertoTotiJ/megaloja-fraud-detection/blob/main/RELATORIO_FINAL_CRISP_DM.md" target="_blank" class="text-info text-decoration-none small font-mono">
                            Ler Relatório Acadêmico Completo (RELATORIO_FINAL_CRISP_DM.md) &rarr;
                        </a>
                    </div>
                </div>
            </div>
        </section>

        <!-- Footer -->
        <footer class="mt-4 pt-3 border-top text-center text-secondary small" style="border-color: var(--color-border) !important;">
            <div class="d-flex flex-wrap justify-content-center gap-4 mb-2">
                <a href="/docs" target="_blank" class="text-secondary text-decoration-none d-inline-flex align-items-center gap-1 hover-white">
                    <!-- Book Open Icon -->
                    <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
                    Swagger Docs (/docs)
                </a>
                <a href="/drift" target="_blank" class="text-secondary text-decoration-none d-inline-flex align-items-center gap-1 hover-white">
                    <!-- Activity Icon -->
                    <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
                    Endpoint Drift (/drift)
                </a>
                <a href="https://github.com/AlbertoTotiJ/megaloja-fraud-detection" target="_blank" class="text-secondary text-decoration-none d-inline-flex align-items-center gap-1 hover-white">
                    <!-- Github Icon -->
                    <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"/><path d="M9 18c-4.51 2-5-2-7-2"/></svg>
                    GitHub Repository
                </a>
            </div>
            <p class="mb-0 font-mono" style="font-size: 0.75rem;">MegaLoja S.A. • FATEC Ourinhos • Projeto Integrador V • 2026</p>
        </footer>

    </main>

    <script>
        // Preset values da Fraude real (Cell 41 do notebook)
        const fraudValues = {
            Time: 406.0, Amount: 149.62,
            V1: -2.31, V2: 1.95, V3: -1.60, V4: 3.99, V5: -0.52, V6: -1.42, V7: -2.53, V8: 1.39,
            V9: -2.77, V10: -2.77, V11: 3.20, V12: -2.89, V13: -0.59, V14: -4.28, V15: 0.38,
            V16: -1.14, V17: -2.83, V18: -0.01, V19: 0.41, V20: 0.12, V21: 0.51, V22: -0.03,
            V23: -0.46, V24: 0.32, V25: 0.04, V26: 0.17, V27: 0.26, V28: -0.14
        };

        const normalValues = {
            Time: 50000.0, Amount: 45.00,
            V1: 1.15, V2: 0.22, V3: 0.35, V4: 0.42, V5: -0.10, V6: 0.05, V7: 0.08, V8: 0.02,
            V9: -0.12, V10: -0.05, V11: 0.15, V12: 0.03, V13: 0.20, V14: 0.10, V15: 0.05,
            V16: 0.01, V17: 0.04, V18: -0.02, V19: 0.05, V20: 0.02, V21: 0.01, V22: 0.02,
            V23: -0.01, V24: 0.04, V25: 0.01, V26: 0.02, V27: 0.01, V28: 0.01
        };

        const suspeitaValues = {
            Time: 11500.0, Amount: 1850.00,
            V1: -1.10, V2: 1.25, V3: -0.85, V4: 1.95, V5: -0.30, V6: -0.65, V7: -1.15, V8: 0.80,
            V9: -1.10, V10: -1.20, V11: 1.45, V12: -1.30, V13: -0.20, V14: -1.90, V15: 0.20,
            V16: -0.70, V17: -1.10, V18: 0.10, V19: 0.30, V20: 0.15, V21: 0.25, V22: -0.10,
            V23: -0.15, V24: 0.10, V25: 0.05, V26: 0.10, V27: 0.12, V28: -0.05
        };

        let currentPCA = { ...fraudValues };

        // Renderiza inputs das variáveis PCA
        function renderizarPCAInputs() {
            const container = document.getElementById('pcaInputsContainer');
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
            const t = parseFloat(document.getElementById('time_val').value) || 0;
            const hour = Math.floor(t / 3600) % 24;
            const min = Math.floor((t % 3600) / 60);
            const sec = Math.floor(t % 60);
            const pad = (n) => String(n).padStart(2, '0');
            document.getElementById('time_hora_display').innerText = `Horário estimado: ${pad(hour)}:${pad(min)}:${pad(sec)} (Hora ${hour})`;
        }

        function carregarPreset(tipo) {
            document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
            if (tipo === 'fraude') {
                document.getElementById('chip-fraude').classList.add('active');
                currentPCA = { ...fraudValues };
            } else if (tipo === 'normal') {
                document.getElementById('chip-normal').classList.add('active');
                currentPCA = { ...normalValues };
            } else if (tipo === 'suspeita') {
                document.getElementById('chip-suspeita').classList.add('active');
                currentPCA = { ...suspeitaValues };
            }

            document.getElementById('time_val').value = currentPCA.Time;
            document.getElementById('amount_val').value = currentPCA.Amount;
            atualizarHoraEstimada();

            for (let i = 1; i <= 28; i++) {
                const el = document.getElementById('pca_v' + i);
                if (el) el.value = currentPCA['V' + i];
            }
        }

        function togglePCA() {
            const grid = document.getElementById('pcaGridContainer');
            const status = document.getElementById('pcaStatusText');
            if (grid.style.display === 'none') {
                grid.style.display = 'block';
                status.innerHTML = 'Recolher &uarr;';
            } else {
                grid.style.display = 'none';
                status.innerHTML = 'Expandir &darr;';
            }
        }

        function switchTab(tabName) {
            document.getElementById('tab-simulador').style.display = tabName === 'simulador' ? 'block' : 'none';
            document.getElementById('tab-drift').style.display = tabName === 'drift' ? 'block' : 'none';
            document.getElementById('tab-crisp').style.display = tabName === 'crisp' ? 'block' : 'none';

            document.getElementById('tab-btn-simulador').classList.toggle('active', tabName === 'simulador');
            document.getElementById('tab-btn-drift').classList.toggle('active', tabName === 'drift');
            document.getElementById('tab-btn-crisp').classList.toggle('active', tabName === 'crisp');
        }

        async function avaliarTransacao() {
            const btn = document.getElementById('btnAvaliar');
            btn.disabled = true;
            btn.innerHTML = `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Avaliando Modelo...`;
            const t0 = performance.now();

            currentPCA.Time = parseFloat(document.getElementById('time_val').value) || 0;
            currentPCA.Amount = parseFloat(document.getElementById('amount_val').value) || 0;

            for (let i = 1; i <= 28; i++) {
                const el = document.getElementById('pca_v' + i);
                if (el) currentPCA['V' + i] = parseFloat(el.value) || 0;
            }

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(currentPCA)
                });
                const data = await response.json();
                const t1 = performance.now();
                const latencyMs = Math.round(t1 - t0);

                const probPercent = (data.fraud_probability * 100).toFixed(2);
                const probElem = document.getElementById('probDisplay');
                probElem.innerText = probPercent + "%";

                const gaugeBar = document.getElementById('gaugeBar');
                gaugeBar.style.width = Math.min(Math.max(probPercent, 5), 100) + "%";

                const banner = document.getElementById('decisaoBanner');
                const titulo = document.getElementById('decisaoTitulo');
                const desc = document.getElementById('decisaoDesc');
                const iconContainer = document.getElementById('decisIncone');

                if (data.fraud_probability >= 0.75) {
                    probElem.style.color = "var(--color-destructive)";
                    gaugeBar.style.backgroundColor = "var(--color-destructive)";
                    banner.style.borderColor = "var(--color-destructive)";
                    banner.style.background = "rgba(239, 68, 68, 0.1)";
                    titulo.className = "h6 mb-1 fw-bold text-danger";
                    titulo.innerText = "BLOQUEIO PREVENTIVO IMEDIATO";
                    desc.innerText = data.recommendation;
                    iconContainer.innerHTML = `<svg class="icon-lg text-danger" viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`;
                } else if (data.is_fraud) {
                    probElem.style.color = "var(--color-warning)";
                    gaugeBar.style.backgroundColor = "var(--color-warning)";
                    banner.style.borderColor = "var(--color-warning)";
                    banner.style.background = "rgba(245, 158, 11, 0.1)";
                    titulo.className = "h6 mb-1 fw-bold text-warning";
                    titulo.innerText = "VERIFICAÇÃO ADICIONAL DE SEGURANÇA (2FA)";
                    desc.innerText = data.recommendation;
                    iconContainer.innerHTML = `<svg class="icon-lg text-warning" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`;
                } else {
                    probElem.style.color = "var(--color-accent)";
                    gaugeBar.style.backgroundColor = "var(--color-accent)";
                    banner.style.borderColor = "var(--color-accent)";
                    banner.style.background = "rgba(34, 197, 94, 0.1)";
                    titulo.className = "h6 mb-1 fw-bold text-success";
                    titulo.innerText = "TRANSAÇÃO APROVADA";
                    desc.innerText = data.recommendation;
                    iconContainer.innerHTML = `<svg class="icon-lg text-success" viewBox="0 0 24 24"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`;
                }

                document.getElementById('metaLatencia').innerText = `Latência: ${latencyMs} ms`;
                document.getElementById('resModeloNome').innerText = data.model_name;
                document.getElementById('resultadoCard').style.display = 'block';
                document.getElementById('resultadoCard').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } catch (err) {
                alert("Erro ao consultar API de predição: " + err);
            } finally {
                btn.disabled = false;
                btn.innerHTML = `<svg class="icon-lg" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg> <span>Avaliar Transação em Tempo Real</span>`;
            }
        }

        async function executarAuditoria(simulateDrift) {
            document.getElementById('driftLoading').style.display = 'block';
            document.getElementById('driftResultadoCard').style.display = 'none';

            try {
                const res = await fetch('/drift?simulate=' + simulateDrift);
                const data = await res.json();

                const alertBox = document.getElementById('alertBoxDrift');
                const title = document.getElementById('alertDriftTitle');
                const msg = document.getElementById('alertDriftMsg');
                const icon = document.getElementById('alertDriftIcon');

                if (data.action_required) {
                    alertBox.style.background = "rgba(239, 68, 68, 0.15)";
                    alertBox.style.borderColor = "var(--color-destructive)";
                    title.className = "h6 mb-1 fw-bold text-danger";
                    title.innerText = "ALERTA DE DEGRAVAÇÃO POPULACIONAL (DATA DRIFT DETECTADO)";
                    msg.className = "mb-0 small text-light";
                    msg.innerText = `${data.recommendation} — (${data.variables_in_drift} de ${data.total_variables_analyzed} variáveis críticas com p < 0.05 / PSI >= 0.20).`;
                    icon.innerHTML = `<svg class="icon-xl text-danger" viewBox="0 0 24 24"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`;
                } else {
                    alertBox.style.background = "rgba(34, 197, 94, 0.15)";
                    alertBox.style.borderColor = "var(--color-accent)";
                    title.className = "h6 mb-1 fw-bold text-success";
                    title.innerText = "OPERAÇÃO ESTÁVEL (SEM DERIVA SIGNIFICATIVA)";
                    msg.className = "mb-0 small text-light";
                    msg.innerText = `${data.recommendation} — Baseline histórica preservada (${data.variables_in_drift} variáveis com desvio leve).`;
                    icon.innerHTML = `<svg class="icon-xl text-success" viewBox="0 0 24 24"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`;
                }

                const tbody = document.getElementById('tabelaDriftBody');
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

                document.getElementById('driftResultadoCard').style.display = 'block';
            } catch (err) {
                alert("Erro ao rodar auditoria de drift: " + err);
            } finally {
                document.getElementById('driftLoading').style.display = 'none';
            }
        }

        // Inicialização
        document.addEventListener('DOMContentLoaded', () => {
            renderizarPCAInputs();
            atualizarHoraEstimada();
        });
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

def get_health_json(request: Request = None):
    res = {
        "status": "online",
        "service": "MegaLoja Fraud Guard API (Vercel Serverless)",
        "model_loaded": pipeline_model is not None,
        "model_name": metadata.get("model_name", "MegaLoja-Fraud-Detector-Bagging-LogisticRegression"),
        "algorithm": metadata.get("algorithm", "BaggingClassifier (base_estimator=LogisticRegression)"),
        "model_version": metadata.get("version", "1.2.0"),
        "drift_monitoring": "Ativo (Kolmogorov-Smirnov & PSI)",
        "documentation": "/docs"
    }
    if request:
        res["debug_headers"] = {k: v for k, v in request.headers.items() if "auth" not in k.lower()}
        res["debug_url"] = str(request.url)
        res["debug_scope_path"] = request.scope.get("path")
    return res

def get_requested_path(request: Request) -> str:
    """Extrai o caminho original da requisição mesmo sob rewrite do Vercel Serverless."""
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
        return JSONResponse(content=get_health_json())

    accept = request.headers.get("accept", "")
    if "application/json" in accept and "text/html" not in accept:
        return JSONResponse(content=get_health_json())
        
    return HTMLResponse(content=HTML_DASHBOARD)

# 2. Endpoint de Health Check
@app.get("/health")
@app.get("/api/health")
def health_endpoint(request: Request):
    return JSONResponse(content=get_health_json(request))

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

# 4. Endpoint de Auditoria de Drift e Degradação (Etapa 5 - Slide 18)
@app.get("/drift")
@app.get("/api/drift")
def drift_endpoint(simulate: bool = False):
    if run_drift_analysis:
        return JSONResponse(content=run_drift_analysis(simulate_drift=simulate))
    return JSONResponse(content={"status": "Drift monitor indisponível"})

# 5. Endpoint de Inferência Preditiva (POST)
@app.post("/predict", response_model=PredictionResponse)
@app.post("/api/predict", response_model=PredictionResponse)
@app.post("/api/index.py/predict", response_model=PredictionResponse)
def predict_endpoint(transaction: TransactionInput):
    res = run_inference(transaction.model_dump())
    return PredictionResponse(**res)

# 6. Roteador Universal Catch-All
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
        return JSONResponse(content=get_health_json())
        
    accept = request.headers.get("accept", "")
    if "application/json" in accept and "text/html" not in accept:
        return JSONResponse(content=get_health_json())
        
    return HTMLResponse(content=HTML_DASHBOARD)
