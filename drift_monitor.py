"""
drift_monitor.py
Módulo de Monitoramento e Manutenção Contínua (Etapa 5 - CRISP-DM / MLOps)
Detecção de Data Drift e Concept Drift com Teste de Kolmogorov-Smirnov (KS-Test) e PSI.
Disciplina: Projeto Integrador V - Prof. Dr. Sidney Carlos Ferrari (FATEC Ourinhos)
Grupo 8: Alberto, Geovane, Gustavo
Agente Responsável: Sentinela (Drift Guardian)
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import ks_2samp

AUDIT_LOG_FILE = "drift_audit_report.json"
MODEL_PATH = "credit_card_fraud_model.joblib"
METADATA_PATH = "model_metadata.json"

def calculate_psi(baseline, target, num_buckets=10):
    """
    Calcula o Population Stability Index (PSI) entre duas distribuições contínuas.
    Regra prática:
    - PSI < 0.10: Sem mudança significativa (estável).
    - 0.10 <= PSI < 0.20: Mudança moderada (requer atenção).
    - PSI >= 0.20: Desvio populacional relevante (gatilho de retreinamento).
    """
    baseline = np.asarray(baseline)
    target = np.asarray(target)
    
    # Cria os limites de quantis a partir da distribuição base
    percentiles = np.linspace(0, 100, num_buckets + 1)
    quantiles = np.percentile(baseline, percentiles)
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf
    
    base_counts, _ = np.histogram(baseline, bins=quantiles)
    target_counts, _ = np.histogram(target, bins=quantiles)
    
    base_pct = base_counts / len(baseline)
    target_pct = target_counts / len(target)
    
    # Evita divisão por zero
    base_pct = np.where(base_pct == 0, 0.0001, base_pct)
    target_pct = np.where(target_pct == 0, 0.0001, target_pct)
    
    psi_val = np.sum((target_pct - base_pct) * np.log(target_pct / base_pct))
    return float(round(psi_val, 4))

def generate_reference_baseline(n_samples=2000, seed=42):
    """Gera distribuição de referência histórica do modelo."""
    np.random.seed(seed)
    data = {
        'Time': np.random.uniform(0, 172800, n_samples),
        'Amount': np.random.exponential(88.0, n_samples),
    }
    data['Hour'] = (data['Time'] / 3600).astype(int) % 24
    for i in range(1, 29):
        data[f'V{i}'] = np.random.normal(0, 1, n_samples)
    return pd.DataFrame(data)

def generate_production_batch(n_samples=500, simulate_drift=False, seed=99):
    """
    Simula um lote de transações colhidas em produção.
    Se simulate_drift=True, simula anomalia comportamental (ex: Black Friday ou ataque massivo).
    """
    np.random.seed(seed)
    if simulate_drift:
        # Mudança drástica no Amount (inflação/Black Friday) e nas componentes de fraude V14/V17
        data = {
            'Time': np.random.uniform(0, 172800, n_samples),
            'Amount': np.random.exponential(350.0, n_samples),  # Média triplicada!
        }
        data['Hour'] = (data['Time'] / 3600).astype(int) % 24
        for i in range(1, 29):
            if i in [4, 12, 14, 17]:
                data[f'V{i}'] = np.random.normal(-2.5, 2.0, n_samples)  # Padrão severamente alterado
            else:
                data[f'V{i}'] = np.random.normal(0, 1, n_samples)
    else:
        # Transações normais aderentes à baseline
        data = {
            'Time': np.random.uniform(0, 172800, n_samples),
            'Amount': np.random.exponential(90.0, n_samples),
        }
        data['Hour'] = (data['Time'] / 3600).astype(int) % 24
        for i in range(1, 29):
            data[f'V{i}'] = np.random.normal(0, 1, n_samples)
            
    return pd.DataFrame(data)

def run_drift_analysis(simulate_drift=False):
    """Executa a auditoria completa de Data Drift e Concept Drift."""
    print("=" * 65)
    print("🛡️  MegaLoja S.A. — MLOps Drift Guardian (Etapa 5 CRISP-DM)")
    print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Modo de Simulação de Drift: {'ATIVADO (Cenário Crítico)' if simulate_drift else 'DESATIVADO (Operação Normal)'}")
    print("=" * 65)
    
    df_baseline = generate_reference_baseline()
    df_production = generate_production_batch(simulate_drift=simulate_drift)
    
    # Features mais críticas no dataset de fraude
    critical_features = ['Amount', 'Hour', 'V4', 'V10', 'V12', 'V14', 'V17', 'V21']
    
    results = {}
    drift_detected_count = 0
    
    print("\n📊 Análise Estatística por Variável:")
    print("-" * 65)
    print(f"{'Feature':<10} | {'KS Stat':<10} | {'p-value':<10} | {'PSI':<8} | {'Status'}")
    print("-" * 65)
    
    for feat in critical_features:
        ks_stat, p_val = ks_2samp(df_baseline[feat], df_production[feat])
        psi_score = calculate_psi(df_baseline[feat], df_production[feat])
        
        # Critério de deriva estatística: p-value < 0.05 ou PSI >= 0.20
        is_drift = (p_val < 0.05) or (psi_score >= 0.20)
        if is_drift:
            drift_detected_count += 1
            status_text = "🚨 DRIFT"
        else:
            status_text = "✅ Estável"
            
        results[feat] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_val), 5),
            "psi": psi_score,
            "drift_detected": bool(is_drift)
        }
        
        print(f"{feat:<10} | {ks_stat:<10.4f} | {p_val:<10.5f} | {psi_score:<8.4f} | {status_text}")
        
    print("-" * 65)
    
    # Decisão de MLOps
    drift_ratio = drift_detected_count / len(critical_features)
    if drift_ratio >= 0.30:  # Se 30% ou mais das variáveis críticas sofreram desvio
        action_required = True
        recommendation = "DISPARO IMEDIATO DE PIPELINE CI/CD: Retreinar modelo com novas transações e recalibrar threshold."
        overall_status = "CRITICAL_DRIFT_DETECTED"
    else:
        action_required = False
        recommendation = "OPERAÇÃO NORMAL: Distribuições aderentes à baseline de treino. Nenhuma intervenção necessária."
        overall_status = "HEALTHY"
        
    print(f"\n📋 DIAGNÓSTICO DO MOTOR DE AUDITORIA:")
    print(f"Status Geral: {overall_status}")
    print(f"Variáveis em Deriva: {drift_detected_count}/{len(critical_features)} ({drift_ratio*100:.1f}%)")
    print(f"Recomendação MLOps: {recommendation}")
    print("=" * 65)
    
    # Salvar Relatório Estruturado de Auditoria (Slide 18)
    audit_report = {
        "report_id": f"drift-audit-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "overall_status": overall_status,
        "action_required": action_required,
        "drift_ratio": round(drift_ratio, 4),
        "variables_in_drift": drift_detected_count,
        "total_variables_analyzed": len(critical_features),
        "recommendation": recommendation,
        "detailed_metrics": results
    }
    
    with open(AUDIT_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=4)
        
    print(f"💾 Relatório persistido com sucesso em: {AUDIT_LOG_FILE}\n")
    return audit_report

if __name__ == '__main__':
    import sys
    simulate = "--simulate-drift" in sys.argv
    run_drift_analysis(simulate_drift=simulate)
