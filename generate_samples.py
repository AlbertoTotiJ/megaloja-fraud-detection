"""
generate_samples.py
Gera e valida as 15 transações de amostra (10 normais e 5 fraudes) para o checkout da MegaLoja.
Mapeia precisamente as 3 reações de negócio:
- Verde (Baixo Risco < 0.50): Pagamento Aprovado
- Amarelo (Médio Risco 0.50 a 0.74): Modal Desafio de Segurança SMS 2FA
- Vermelho (Alto Risco >= 0.75 / Fraude): Transação Negada por Política de Segurança
"""
import json
import joblib
import pandas as pd
import numpy as np

model = joblib.load("credit_card_fraud_model.joblib")
with open("model_metadata.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

expected_features = meta["features"]

# 1. 5 FRAUDES REAIS (Cell 41 e padrões severos de fraude)
frauds = [
    {
        "id": "TX-FRAUD-001",
        "description": "Fraude #1: Clonagem Noturna Confirmada (Amostra Cell 41)",
        "expected_category": "fraud",
        "category_label": "Fraude Confirmada",
        "Time": 406.0, "Amount": 149.62,
        "V1": -2.31, "V2": 1.95, "V3": -1.60, "V4": 3.99, "V5": -0.52, "V6": -1.42, "V7": -2.53, "V8": 1.39,
        "V9": -2.77, "V10": -2.77, "V11": 3.20, "V12": -2.89, "V13": -0.59, "V14": -4.28, "V15": 0.38,
        "V16": -1.14, "V17": -2.83, "V18": -0.01, "V19": 0.41, "V20": 0.12, "V21": 0.51, "V22": -0.03,
        "V23": -0.46, "V24": 0.32, "V25": 0.04, "V26": 0.17, "V27": 0.26, "V28": -0.14
    },
    {
        "id": "TX-FRAUD-002",
        "description": "Fraude #2: Compra Abusiva com Cartão Furtado (03:15h)",
        "expected_category": "fraud",
        "category_label": "Fraude Confirmada",
        "Time": 11800.0, "Amount": 4899.00,
        "V1": -3.05, "V2": 2.50, "V3": -2.90, "V4": 4.10, "V5": -1.20, "V6": -1.80, "V7": -3.10, "V8": 1.80,
        "V9": -3.20, "V10": -3.50, "V11": 3.80, "V12": -3.90, "V13": -0.80, "V14": -5.10, "V15": 0.40,
        "V16": -1.80, "V17": -3.90, "V18": -0.20, "V19": 0.60, "V20": 0.40, "V21": 0.80, "V22": -0.10,
        "V23": -0.50, "V24": 0.20, "V25": 0.10, "V26": 0.30, "V27": 0.40, "V28": -0.20
    },
    {
        "id": "TX-FRAUD-003",
        "description": "Fraude #3: Injeção de Cartão Bloqueado no Emissor",
        "expected_category": "fraud",
        "category_label": "Fraude Confirmada",
        "Time": 472.0, "Amount": 529.00,
        "V1": -2.80, "V2": 2.10, "V3": -2.20, "V4": 3.75, "V5": -0.90, "V6": -1.60, "V7": -2.80, "V8": 1.50,
        "V9": -2.95, "V10": -3.10, "V11": 3.40, "V12": -3.20, "V13": -0.70, "V14": -4.60, "V15": 0.25,
        "V16": -1.40, "V17": -3.20, "V18": -0.10, "V19": 0.50, "V20": 0.25, "V21": 0.65, "V22": -0.05,
        "V23": -0.48, "V24": 0.25, "V25": 0.08, "V26": 0.22, "V27": 0.32, "V28": -0.18
    },
    {
        "id": "TX-FRAUD-004",
        "description": "Fraude #4: Ataque Automatizado de Teste de Cartões (Botnet)",
        "expected_category": "fraud",
        "category_label": "Fraude Confirmada",
        "Time": 8500.0, "Amount": 1290.00,
        "V1": -2.55, "V2": 2.05, "V3": -1.95, "V4": 3.85, "V5": -0.75, "V6": -1.50, "V7": -2.65, "V8": 1.45,
        "V9": -2.85, "V10": -2.95, "V11": 3.30, "V12": -3.05, "V13": -0.65, "V14": -4.45, "V15": 0.32,
        "V16": -1.25, "V17": -3.05, "V18": -0.05, "V19": 0.45, "V20": 0.18, "V21": 0.58, "V22": -0.04,
        "V23": -0.47, "V24": 0.28, "V25": 0.06, "V26": 0.19, "V27": 0.29, "V28": -0.16
    },
    {
        "id": "TX-FRAUD-005",
        "description": "Fraude #5: Golpe de Alto Valor via Proxy Anônimo",
        "expected_category": "fraud",
        "category_label": "Fraude Confirmada",
        "Time": 24000.0, "Amount": 8999.00,
        "V1": -3.40, "V2": 2.80, "V3": -3.10, "V4": 4.50, "V5": -1.50, "V6": -2.00, "V7": -3.40, "V8": 2.00,
        "V9": -3.50, "V10": -3.80, "V11": 4.10, "V12": -4.20, "V13": -0.90, "V14": -5.50, "V15": 0.45,
        "V16": -2.00, "V17": -4.20, "V18": -0.30, "V19": 0.70, "V20": 0.50, "V21": 0.90, "V22": -0.15,
        "V23": -0.55, "V24": 0.15, "V25": 0.12, "V26": 0.35, "V27": 0.45, "V28": -0.25
    }
]

# 2. 10 TRANSAÇÕES NORMAIS
# 7 Verde (Aprovado direto < 0.50) + 3 Amarelo (Zona Amarela / Modal SMS 2FA 0.50 a 0.74)
normals = [
    # 7 Verdes: Pagamento Aprovado
    {
        "id": "TX-NORM-001",
        "description": "Compra Normal #1: Pagamento habitual em horário comercial",
        "expected_category": "normal",
        "category_label": "Compra Legítima",
        "Time": 45000.0, "Amount": 89.90,
        "V16": 1.6, "V25": 1.4, "V7": 1.3, "V26": 1.1, "V12": 1.2, "V23": 0.9,
        "V15": -0.6, "V5": -0.5, "V8": -0.6, "V21": -0.5
    },
    {
        "id": "TX-NORM-002",
        "description": "Compra Normal #2: E-commerce recorrente com cliente VIP",
        "expected_category": "normal",
        "category_label": "Compra Legítima",
        "Time": 54000.0, "Amount": 149.00,
        "V16": 1.8, "V25": 1.6, "V7": 1.4, "V26": 1.2, "V12": 1.3, "V23": 1.0,
        "V15": -0.7, "V5": -0.6, "V8": -0.7, "V21": -0.6
    },
    {
        "id": "TX-NORM-003",
        "description": "Compra Normal #3: Livro técnico parcelado sem juros",
        "expected_category": "normal",
        "category_label": "Compra Legítima",
        "Time": 62000.0, "Amount": 79.50,
        "V16": 1.5, "V25": 1.3, "V7": 1.1, "V26": 1.0, "V12": 1.1, "V23": 0.8,
        "V15": -0.5, "V5": -0.4, "V8": -0.5, "V21": -0.4
    },
    {
        "id": "TX-NORM-004",
        "description": "Compra Normal #4: Periféricos de informática com entrega expressa",
        "expected_category": "normal",
        "category_label": "Compra Legítima",
        "Time": 39000.0, "Amount": 249.90,
        "V16": 1.4, "V25": 1.2, "V7": 1.0, "V26": 0.9, "V12": 1.0, "V23": 0.7,
        "V15": -0.4, "V5": -0.3, "V8": -0.4, "V21": -0.3
    },
    {
        "id": "TX-NORM-005",
        "description": "Compra Normal #5: Acessórios para celular na MegaLoja",
        "expected_category": "normal",
        "category_label": "Compra Legítima",
        "Time": 48000.0, "Amount": 45.00,
        "V16": 1.7, "V25": 1.5, "V7": 1.3, "V26": 1.1, "V12": 1.2, "V23": 0.9,
        "V15": -0.6, "V5": -0.5, "V8": -0.6, "V21": -0.5
    },
    {
        "id": "TX-NORM-006",
        "description": "Compra Normal #6: Monitor gamer com frete grátis",
        "expected_category": "normal",
        "category_label": "Compra Legítima",
        "Time": 51000.0, "Amount": 350.00,
        "V16": 1.5, "V25": 1.3, "V7": 1.2, "V26": 1.0, "V12": 1.1, "V23": 0.8,
        "V15": -0.5, "V5": -0.4, "V8": -0.5, "V21": -0.4
    },
    {
        "id": "TX-NORM-007",
        "description": "Compra Normal #7: Fone Bluetooth com pagamento à vista",
        "expected_category": "normal",
        "category_label": "Compra Legítima",
        "Time": 58000.0, "Amount": 199.90,
        "V16": 1.9, "V25": 1.7, "V7": 1.5, "V26": 1.3, "V12": 1.4, "V23": 1.1,
        "V15": -0.8, "V5": -0.7, "V8": -0.8, "V21": -0.7
    },
    # 3 Amarelas (Zona Amarela: Risco Moderado -> Desafio 2FA SMS entre 0.50 e 0.74)
    {
        "id": "TX-NORM-008",
        "description": "Zona Amarela #1: Compra noturna de celular novo (Requer SMS 2FA)",
        "expected_category": "yellow_zone",
        "category_label": "Verificação 2FA",
        "Time": 65000.0, "Amount": 380.00,
        "V16": 0.78, "V25": 0.65, "V7": 0.45, "V26": 0.40, "V12": 0.45, "V23": 0.30,
        "V15": -0.10, "V5": -0.05, "V8": -0.10, "V21": -0.05
    },
    {
        "id": "TX-NORM-009",
        "description": "Zona Amarela #2: Novo endereço de entrega cadastrado (Requer SMS 2FA)",
        "expected_category": "yellow_zone",
        "category_label": "Verificação 2FA",
        "Time": 65000.0, "Amount": 410.00,
        "V16": 0.76, "V25": 0.65, "V7": 0.45, "V26": 0.40, "V12": 0.45, "V23": 0.30,
        "V15": -0.10, "V5": -0.05, "V8": -0.10, "V21": -0.05
    },
    {
        "id": "TX-NORM-010",
        "description": "Zona Amarela #3: Transação acima da média habitual (Requer SMS 2FA)",
        "expected_category": "yellow_zone",
        "category_label": "Verificação 2FA",
        "Time": 65000.0, "Amount": 450.00,
        "V16": 0.75, "V25": 0.65, "V7": 0.45, "V26": 0.40, "V12": 0.45, "V23": 0.30,
        "V15": -0.10, "V5": -0.05, "V8": -0.10, "V21": -0.05
    }
]

# Completa variáveis faltantes com 0.0
for n in normals:
    for i in range(1, 29):
        if f"V{i}" not in n:
            n[f"V{i}"] = 0.0

all_samples = normals + frauds

print("=" * 80)
print("AUDITORIA DAS 15 TRANSAÇÕES DA AMOSTRA DE CHECKOUT")
print("=" * 80)
print(f"{'ID':<12} | {'Categoria':<14} | {'Valor (R$)':<10} | {'Score':<8} | {'Status Visual Esperado'}")
print("-" * 80)

for s in all_samples:
    df_row = pd.DataFrame([s])
    df_row["Hour"] = (df_row["Time"] / 3600).astype(int) % 24
    df_row = df_row[expected_features]
    prob = float(model.predict_proba(df_row)[0, 1])
    s["fraud_probability"] = round(prob, 4)
    
    if prob >= 0.75:
        s["risk_zone"] = "danger"
        s["visual_action"] = "BLOCKED_RED"
        s["action_message"] = "Transação Negada por Política de Segurança"
        badge = "🔴 VERMELHO (Transação Negada)"
    elif prob >= 0.50:
        s["risk_zone"] = "warning"
        s["visual_action"] = "CHALLENGE_YELLOW"
        s["action_message"] = "Atenção: Insira o código SMS enviado ao seu celular"
        badge = "🟡 AMARELO (Modal SMS 2FA)"
    else:
        s["risk_zone"] = "success"
        s["visual_action"] = "APPROVED_GREEN"
        s["action_message"] = "Pagamento Aprovado"
        badge = "🟢 VERDE (Pagamento Aprovado)"
        
    print(f"{s['id']:<12} | {s['category_label']:<14} | R$ {s['Amount']:<8.2f} | {s['fraud_probability']*100:<5.1f}% | {badge}")

print("=" * 80)

with open("transactions_sample.json", "w", encoding="utf-8") as f:
    json.dump(all_samples, f, indent=2, ensure_ascii=False)

with open("api/transactions_sample.json", "w", encoding="utf-8") as f:
    json.dump(all_samples, f, indent=2, ensure_ascii=False)

print("\nArquivo 'transactions_sample.json' e 'api/transactions_sample.json' gravados com sucesso!")
