"""
app_dashboard.py
Interface Web Interativa em Streamlit para Operações e MLOps — MegaLoja Fraud Guard.
Alinhado com o Design System Dark OLED (Fira Code/Sans) e Metodologia CRISP-DM.
Grupo 8 - PI V FATEC Ourinhos (Prof. Dr. Sidney Carlos Ferrari)
"""
import streamlit as st
import requests
import json
import pandas as pd
import numpy as np

# Configuração da página
st.set_page_config(
    page_title="MegaLoja Fraud Guard — Dashboard Executivo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Fira Sans', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6, [data-testid="stMetricValue"] {
        font-family: 'Fira Code', monospace !important;
        letter-spacing: -0.02em;
    }

    .stButton>button {
        background-color: #22C55E !important;
        color: #0F172A !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 200ms ease !important;
    }
    .stButton>button:hover {
        background-color: #16A34A !important;
        box-shadow: 0 0 15px rgba(34, 197, 94, 0.3) !important;
        transform: translateY(-1px);
    }

    .card-banner {
        background: #0E1223;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="card-banner">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
        <div>
            <h2 style="margin: 0; color: #FFFFFF; font-weight: 700;">MegaLoja Fraud Guard</h2>
            <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.9rem;">
                Sistema Preditivo em Tempo Real & Monitoramento MLOps (Metodologia CRISP-DM)
            </p>
        </div>
        <div>
            <span style="background: rgba(34, 197, 94, 0.15); color: #4ADE80; border: 1px solid rgba(34, 197, 94, 0.3); padding: 4px 10px; border-radius: 6px; font-family: 'Fira Code', monospace; font-size: 0.8rem; font-weight: 600;">
                v1.2.0 • ONLINE
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### ⚙️ Configurações de Nuvem")
api_url = st.sidebar.text_input("Endpoint da API:", "https://megaloja-fraud-detection.vercel.app/predict")
st.sidebar.caption("Conectado à API Serverless na nuvem Vercel (< 15ms de latência)")

st.sidebar.markdown("---")
st.sidebar.markdown("### 👥 Equipe do Projeto (Grupo 8)")
st.sidebar.markdown("- **Alberto**\n- **Geovane**\n- **Gustavo**")
st.sidebar.markdown("---")
st.sidebar.caption("Projeto Integrador V • FATEC Ourinhos\nProf. Dr. Sidney Carlos Ferrari")

# Tabs
tab1, tab2, tab3 = st.tabs(["1. Inferência em Tempo Real", "2. Monitoramento de Data Drift (Etapa 5)", "3. Governança CRISP-DM"])

with tab1:
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.subheader("Simulação de Transação")
        
        tipo_simulacao = st.radio(
            "Cenários Pré-configurados:",
            ["Fraude Crítica (Cell 41 do Notebook)", "Compra Cotidiana Segura (R$ 45,00)", "Desafio de Segurança / 2FA (R$ 1.850,00)"]
        )
        
        if "Fraude Crítica" in tipo_simulacao:
            def_time = 406.0
            def_amount = 149.62
            pca_defaults = {
                1: -2.31, 2: 1.95, 3: -1.60, 4: 3.99, 5: -0.52, 6: -1.42, 7: -2.53, 8: 1.39,
                9: -2.77, 10: -2.77, 11: 3.20, 12: -2.89, 13: -0.59, 14: -4.28, 15: 0.38,
                16: -1.14, 17: -2.83, 18: -0.01, 19: 0.41, 20: 0.12, 21: 0.51, 22: -0.03,
                23: -0.46, 24: 0.32, 25: 0.04, 26: 0.17, 27: 0.26, 28: -0.14
            }
        elif "Compra Cotidiana" in tipo_simulacao:
            def_time = 50000.0
            def_amount = 45.00
            pca_defaults = {i: 0.02 for i in range(1, 29)}
            pca_defaults[1] = 1.15
            pca_defaults[2] = 0.22
            pca_defaults[3] = 0.35
        else:
            def_time = 11500.0
            def_amount = 1850.00
            pca_defaults = {i: 0.05 for i in range(1, 29)}
            pca_defaults[1] = -1.10
            pca_defaults[4] = 1.95
            pca_defaults[14] = -1.90
            
        time_val = st.number_input("Tempo Decorrido (segundos):", value=def_time, step=10.0)
        hour_calc = int(time_val / 3600) % 24
        st.caption(f"Horário estimado da transação: {hour_calc:02d}:00h")
        
        amount_val = st.number_input("Valor da Compra (R$):", value=def_amount, step=1.0)
        
        with st.expander("Componentes Principais Criptografados (PCA V1 a V28)"):
            pca_cols = st.columns(4)
            v_values = {}
            for i in range(1, 29):
                col_target = pca_cols[(i - 1) % 4]
                v_values[f"V{i}"] = col_target.number_input(f"V{i}", value=float(pca_defaults.get(i, 0.0)), step=0.1, key=f"pca_input_{i}")

    with col2:
        st.subheader("Parecer do Motor de Risco")
        
        if st.button("Avaliar Transação em Tempo Real", use_container_width=True):
            payload = {"Time": time_val, "Amount": amount_val, **v_values}
            
            try:
                with st.spinner("Consultando modelo preditivo na nuvem Vercel..."):
                    response = requests.post(api_url, json=payload, timeout=5)
                    
                if response.status_code == 200:
                    result = response.json()
                    prob = result["fraud_probability"] * 100
                    is_fraud = result["is_fraud"]
                    rec = result["recommendation"]
                    
                    st.metric(
                        label="Probabilidade Calculada de Fraude",
                        value=f"{prob:.2f}%",
                        delta="ALTO RISCO" if is_fraud else "SEGURO",
                        delta_color="inverse" if is_fraud else "normal"
                    )
                    
                    if prob >= 75:
                        st.error(f"**PARECER:** {rec}")
                    elif is_fraud:
                        st.warning(f"**PARECER:** {rec}")
                    else:
                        st.success(f"**PARECER:** {rec}")
                        
                    st.progress(int(min(max(prob, 0), 100)))
                    
                    with st.expander("Telemetria & Resposta JSON Completa"):
                        st.json(result)
                else:
                    st.error(f"Erro retornado pela API ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Falha de conexão com `{api_url}`: {e}")

with tab2:
    st.subheader("Auditoria Contínua de Data Drift (Etapa 5 - Slide 18)")
    st.markdown("""
    Modelos preditivos sofrem degradação por **Data Drift** (mudança na distribuição de atributos) e **Concept Drift** (mudança no padrão da fraude).
    A auditoria executa o **Teste de Kolmogorov-Smirnov (KS-Test)** e o **Population Stability Index (PSI)** sobre as variáveis críticas.
    """)
    
    col_drift1, col_drift2 = st.columns(2)
    
    with col_drift1:
        if st.button("Auditar Operação Normal (Baseline Estável)", use_container_width=True):
            try:
                import drift_monitor
                report = drift_monitor.run_drift_analysis(simulate_drift=False)
                st.success(f"**Status MLOps:** {report['overall_status']}")
                st.info(f"**Recomendação:** {report['recommendation']}")
                df_drift = pd.DataFrame(report['detailed_metrics']).T
                st.dataframe(df_drift, use_container_width=True)
            except Exception as e:
                st.error(f"Erro ao executar auditoria: {e}")
                
    with col_drift2:
        if st.button("Simular Anomalia / Black Friday (Drift Crítico)", use_container_width=True):
            try:
                import drift_monitor
                report = drift_monitor.run_drift_analysis(simulate_drift=True)
                st.error(f"**Status MLOps:** {report['overall_status']}")
                st.warning(f"**Recomendação:** {report['recommendation']}")
                df_drift = pd.DataFrame(report['detailed_metrics']).T
                st.dataframe(df_drift, use_container_width=True)
            except Exception as e:
                st.error(f"Erro ao simular drift: {e}")

with tab3:
    st.subheader("Governança e Metodologia CRISP-DM")
    st.markdown("""
    - **Etapa 1: Entendimento do Negócio:** Diagnóstico das perdas por fraude na MegaLoja S.A.
    - **Etapa 2: Entendimento & Preparação dos Dados:** Under-sampling balanceado (Cell 41) e extração temporal (`Hour`).
    - **Etapa 3: Modelagem Preditiva:** Bagging de 10 Regressões Logísticas garantindo convergência e interpretabilidade.
    - **Etapa 4: Avaliação & Deploy:** FastAPI em Vercel Serverless com latência inferior a 15ms.
    - **Etapa 5: Monitoramento & MLOps:** KS-Test e PSI contínuos com gatilho automático de retreinamento.
    """)
