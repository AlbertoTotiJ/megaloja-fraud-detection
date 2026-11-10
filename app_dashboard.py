"""
app_dashboard.py
Interface Web interativa em Streamlit para simulação executiva de transações.
Grupo 8 - PI V - FATEC Ourinhos (Prof. Dr. Sidney Carlos Ferrari)
"""
import streamlit as st
import requests
import json
import numpy as np

st.set_page_config(
    page_title="Painel de Fraudes - MegaLoja S.A.",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ MegaLoja S.A. — Sistema Inteligente de Detecção de Fraudes")
st.markdown("""
Painel operacional para análise preditiva de transações com cartões de crédito.
*Projeto Integrador V — FATEC Ourinhos | Prof. Dr. Sidney Carlos Ferrari*
""")

st.sidebar.header("⚙️ Configurações da API")
api_url = st.sidebar.text_input("Endpoint da API:", "https://megaloja-fraud-detection.vercel.app/predict")
st.sidebar.caption("💡 Conectado à API Serverless na nuvem Vercel")

st.sidebar.markdown("---")
st.sidebar.markdown("### 👥 Equipe do Projeto (Grupo 8)")
st.sidebar.markdown("- Alberto\n- Geovane\n- Gustavo")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Dados da Transação")
    time_val = st.number_input("Tempo (segundos transcorridos):", value=406.0, step=10.0)
    amount_val = st.number_input("Valor da Compra (R$):", value=149.62, step=1.0)
    
    st.markdown("##### Variáveis Criptografadas (PCA Components)")
    tipo_simulacao = st.radio("Selecione o cenário de teste:", ["Transação Fraudulenta (Amostra Real)", "Transação Normal (Amostra Típica)"])
    
    defaults = {}
    if tipo_simulacao == "Transação Fraudulenta (Amostra Real)":
        defaults = {
            1: -2.31, 2: 1.95, 3: -1.60, 4: 3.99, 5: -0.52, 6: -1.42, 7: -2.53, 8: 1.39,
            9: -2.77, 10: -2.77, 11: 3.20, 12: -2.89, 13: -0.59, 14: -4.28, 15: 0.38,
            16: -1.14, 17: -2.83, 18: -0.01, 19: 0.41, 20: 0.12, 21: 0.51, 22: -0.03,
            23: -0.46, 24: 0.32, 25: 0.04, 26: 0.17, 27: 0.26, 28: -0.14
        }
    else:
        defaults = {i: 0.05 for i in range(1, 29)}
        defaults[1] = 1.15
        defaults[2] = 0.22
        defaults[3] = 0.35
        
    v_values = {}
    for i in range(1, 29):
        v_values[f"V{i}"] = defaults.get(i, 0.0)

with col2:
    st.subheader("📊 Resultado da Avaliação de Risco")
    
    if st.button("🔍 Avaliar Transação em Tempo Real", use_container_width=True):
        payload = {"Time": time_val, "Amount": amount_val, **v_values}
        
        try:
            with st.spinner("Consultando modelo em produção..."):
                response = requests.post(api_url, json=payload, timeout=5)
                
            if response.status_code == 200:
                result = response.json()
                prob = result["fraud_probability"] * 100
                is_fraud = result["is_fraud"]
                rec = result["recommendation"]
                
                st.metric(
                    label="Probabilidade Calculada de Fraude",
                    value=f"{prob:.2f}%",
                    delta=f"{'ALTO RISCO' if is_fraud else 'SEGURO'}",
                    delta_color="inverse" if is_fraud else "normal"
                )
                
                if is_fraud:
                    st.error(f"🚨 **DECISÃO DO MOTOR:** {rec}")
                else:
                    st.success(f"✅ **DECISÃO DO MOTOR:** {rec}")
                    
                st.progress(int(min(max(prob, 0), 100)))
                
                with st.expander("Ver Resposta JSON Completa da API"):
                    st.json(result)
            else:
                st.error(f"Erro retornado pela API: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Não foi possível conectar à API em `{api_url}`. Verifique se o serviço está ativo. Detalhes: {e}")

st.markdown("---")
st.subheader("📈 Etapa 5: Monitoramento Contínuo e Data Drift (CRISP-DM / MLOps)")
st.caption("Implementação da Etapa 5 descrita no Slide 18 da aula do Prof. Dr. Sidney Carlos Ferrari.")

col_drift1, col_drift2 = st.columns([1, 1])

with col_drift1:
    if st.button("📊 Auditar Data Drift (Operação Normal)", use_container_width=True):
        try:
            import drift_monitor
            report = drift_monitor.run_drift_analysis(simulate_drift=False)
            st.success(f"**Status MLOps:** {report['overall_status']}")
            st.info(f"**Recomendação:** {report['recommendation']}")
            df_drift = pd.DataFrame(report['detailed_metrics']).T
            st.dataframe(df_drift, use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao rodar auditoria local: {e}")

with col_drift2:
    if st.button("⚠️ Simular Anomalia / Black Friday (Gatilho de Retreinamento)", use_container_width=True):
        try:
            import drift_monitor
            report = drift_monitor.run_drift_analysis(simulate_drift=True)
            st.error(f"**Status MLOps:** {report['overall_status']}")
            st.warning(f"**Recomendação:** {report['recommendation']}")
            df_drift = pd.DataFrame(report['detailed_metrics']).T
            st.dataframe(df_drift, use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao simular drift: {e}")
