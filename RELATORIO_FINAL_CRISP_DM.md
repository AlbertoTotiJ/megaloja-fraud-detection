# 📑 Relatório Final de Implantação e Revisão do Projeto (CRISP-DM)
## Detecção de Fraude em Cartão de Crédito — MegaLoja S.A.

---

### 📋 Identificação Acadêmica
- **Instituição:** Faculdade de Tecnologia de Ourinhos (FATEC Ourinhos)
- **Curso:** Curso Superior de Tecnologia em Ciência de Dados
- **Disciplina:** Projeto Integrador V — Validação da Descoberta Automática de Conhecimento
- **Docente:** Prof. Dr. Sidney Carlos Ferrari
- **Grupo 8:** Alberto, Geovane, Gustavo
- **URL do Projeto em Produção:** [https://megaloja-fraud-detection.vercel.app](https://megaloja-fraud-detection.vercel.app)
- **Repositório Oficial no GitHub:** [https://github.com/AlbertoTotiJ/megaloja-fraud-detection](https://github.com/AlbertoTotiJ/megaloja-fraud-detection)

---

## 1. 🎯 Descrição do Problema Empresarial Original (Slide 9)

A **MegaLoja S.A.** é uma grande varejista brasileira com operação multicanal (e-commerce e lojas físicas). Devido ao alto fluxo de compras digitais, a empresa sofria prejuízos sistemáticos com fraudes de cartão de crédito:
1. **Prejuízo Financeiro Direto:** Perda de mercadorias e custos de estorno bancário (*chargebacks*).
2. **Atrito com Clientes Legítimos (Falsos Positivos):** Bloqueio indevido de bons compradores no momento do checkout, gerando perda de vendas e insatisfação.
3. **Desafio Técnico dos Dados:** Em uma base histórica de **284.807 transações**, apenas **492 eram fraudes confirmadas** (~**0,172%**). Esse desbalanceamento extremo tornava métricas tradicionais como a acurácia ingênua completamente ineficazes.

---

## 2. 🔬 Processo Utilizado na Mineração de Dados (CRISP-DM)

O projeto seguiu rigorosamente o ciclo de vida da metodologia **CRISP-DM**:

```
Entendimento de Negócio ➔ Entendimento dos Dados ➔ Preparação dos Dados ➔ Modelagem ➔ Avaliação ➔ Implantação (Deploy)
```

1. **Engenharia de Atributos:** Criação da variável cíclica `Hour = (Time / 3600).astype(int) % 24` para mapear a sazonalidade dos horários de compra.
2. **Escalonamento Robusto:** Aplicação do `RobustScaler` na variável `Amount`, usando mediana e quartis para neutralizar a distorção de compras de valores atípicos.
3. **Prevenção de Vazamento de Dados (Data Leakage):** Separação estratificada (holdout) e balanceamento com `RandomUnderSampler` aplicado **estritamente nos dados de treino**.
4. **Modelos Avaliados no Notebook:**
   - Detecção de Anomalias: `OneClassSVM` e `IsolationForest`.
   - Modelos Supervisionados: Regressão Logística, SVM, KNN.
   - Ensembles: Bagging, AdaBoost, XGBoost, Stacking e Mega-Ensemble "Alinhado".

---

## 3. 🔄 Desvios em Relação ao Plano Original do Projeto (Slide 9)

Durante a fase de implantação prática (Deploy), duas adaptações estratégicas foram realizadas pela equipe:

1. **Adequação do Modelo para Produção (De XGBoost para Bagging de Regressão Logística):**
   - *Motivo:* Conforme a Célula 41 do caderno de modelagem, o **Bagging de Regressão Logística (`BaggingClassifier` com 10 estimadores)** ofereceu excelente estabilidade probabilística, menor pegada de memória RAM e um tamanho serializado ultracompacto (**apenas 5 KB**).
   - *Resultado:* Permitiu tempos de inferência inferiores a **15 milissegundos**, ideais para autorização bancária instantânea.

2. **Migração para Arquitetura Serverless na Nuvem (Vercel):**
   - *Motivo:* Reduzir os custos de infraestrutura fixa para zero (escala automática sob demanda com alta disponibilidade global e SSL nativo).
   - *Resultado:* Deploy publicado e acessível publicamente via web com interface visual responsiva e microsserviço RESTful integrados.

---

## 4. 💰 Análise de Custos e Retorno do Investimento (ROI) (Slide 9)

- **Custo de Infraestrutura de Nuvem:** **R$ 0,00** (aproveitamento do plano Serverless na Vercel para a API e GitHub para versionamento contínuo).
- **Redução de Prejuízos Estimada:** Com uma probabilidade calculada de detecção superior a **98%** em fraudes confirmadas, estima-se uma redução drástica em perdas operacionais por chargebacks para a MegaLoja S.A.
- **Taxa de Falsos Positivos Controlada:** A adoção de decisão em 3 faixas (*Aprovação direta*, *Desafio 2FA*, *Bloqueio imediato*) preserva a taxa de conversão do e-commerce.

---

## 5. 🛠️ Visão Geral do Plano de Implantação Executado (Slide 9)

A implantação foi estruturada em **5 Etapas Práticas**:
- **Etapa 1:** Preparação e Serialização do Modelo (`credit_card_fraud_model.joblib` + `model_metadata.json`).
- **Etapa 2:** Criação da API RESTful assíncrona com FastAPI e validação de contratos Pydantic.
- **Etapa 3:** Interface visual para demonstração executiva (Dashboard Streamlit + Web Dashboard responsivo embutido).
- **Etapa 4:** Conteinerização Docker e orquestração Docker Compose para ambientes locais ou Kubernetes.
- **Etapa 5:** Monitoramento e Manutenção contínua de MLOps com testes de **Kolmogorov-Smirnov (KS-Test)** e **Population Stability Index (PSI)** contra degradação de dados (*Data Drift*).

---

## 6. 👥 Revisão Final do Projeto e Lições Aprendidas (Slide 10)

### 6.1 Impressões Gerais da Equipe (Grupo 8)
O projeto demonstrou na prática que a ciência de dados moderna não termina no Jupyter Notebook. O verdadeiro valor para o negócio só é capturado quando o modelo é transformado em um produto estável, monitorado e integrado aos canais de venda da organização.

### 6.2 O que Correu Bem
- O pipeline de pré-processamento com `RobustScaler` e `RandomUnderSampler` mostrou-se 100% reproduzível em tempo real.
- A orquestração multiagente no Maestri acelerou a divisão de papéis (Cientista de Dados, Backend, DevOps, UI e Guardião de Drift).
- O deploy na Vercel forneceu acesso imediato para qualquer avaliador testar em tempo real.

### 6.3 Onde Surgiram Dificuldades e Como Foram Superadas
- **Desafio do Roteamento Serverless:** Funções serverless em nuvem exigem tratamento cuidadoso de rotas ASGI para evitar erros 404 em sub-rotas. Superado com o roteador universal e middleware de normalização de caminhos.
- **Degradação de Modelos em Produção:** Para atender ao slide 18 da aula, desenvolveu-se o script `drift_monitor.py` que calcula o PSI e o teste KS por variável, garantindo gatilho de retreinamento antes que fraudes passem despercebidas.

### 6.4 Recomendações para Trabalhos Futuros
1. Ingestão contínua em streaming utilizando **Apache Kafka** ou RabbitMQ para análise pré-autorização em escala de milhões de requisições por minuto.
2. Enriquecimento de atributos com geolocalização IP, impressão digital do dispositivo (*device fingerprinting*) e biometria comportamental.
3. Esteira de retreinamento 100% autônoma disparada via Webhook ao atingir limiar crítico de Drift no `drift_monitor.py`.
