"""
Indicador de Paz - Versão Deploy-Ready com Streamlit (A3)

Este script é um aplicativo Streamlit que implementa um pipeline ETL mínimo para calcular
um "Indicador de Paz (IP)" por país. O app inclui:
- Modo de demonstração (dados sintéticos) para rodar imediatamente;
- Ganchos (funções) para baixar dados reais de UCDP, UNODC, SIPRI, World Bank e UNHCR;
  - Observação: ACLED normalmente exige registro e token; deixei espaço para inserir o token.
- Harmonização básica, normalização (min-max worst-is-high), agregação por pesos e cálculo do IP (0-100);
- Interface para ajustar pesos, escolher ano (quando dados reais estiverem disponíveis), e exportar CSV;
- Visualizações: tabela, gráfico de barras, e mapa coroplético (Plotly).

Como usar:
1. Instale dependências:
   pip install streamlit pandas numpy plotly requests pycountry
2. Rode localmente:
   streamlit run indicador_paz_streamlit.py
3. No app, você pode usar o modo DEMO para testar imediatamente;
   para dados reais, utilize as funções de download (devem ser executadas no seu ambiente com acesso à internet).

Observações importantes:
- As URLs de download e APIs podem alterar com o tempo; verifique as fontes oficiais.
- Os downloads em "modo real" tentam buscar arquivos CSV públicos; se falharem, o app oferecerá usar os dados sintéticos.
- Ajuste os pesos conforme sua metodologia.

"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import os
import pycountry
import plotly.express as px

# ------------------------- Configurações iniciais -------------------------
st.set_page_config(page_title="Indicador de Paz", layout="wide")
st.title("Indicador de Paz — Protótipo (Streamlit)")

# ------------------------- Utilitários -------------------------

def minmax_worse_is_high(series: pd.Series) -> pd.Series:
    if series.max() == series.min():
        return pd.Series(0.0, index=series.index)
    return (series - series.min()) / (series.max() - series.min())


def iso3_from_country(name: str) -> str:
    try:
        return pycountry.countries.lookup(name).alpha_3
    except Exception:
        return None

# ------------------------- Dados de exemplo (modo demo) -------------------------

def sample_data():
    data = {
        "country": ["Norway", "Sweden", "Brazil", "South Africa", "Ukraine", "Afghanistan", "USA", "India", "Mexico", "Syria"],
        "homicide_per_100k": [0.5, 1.0, 27.0, 35.9, 6.5, 5.2, 5.0, 3.2, 29.0, 12.0],
        "battle_deaths_per_100k": [0.0, 0.0, 0.1, 0.2, 45.0, 120.0, 0.5, 0.1, 0.2, 300.0],
        "conflict_event_rate": [0.1, 0.2, 3.0, 5.5, 80.0, 200.0, 2.0, 1.5, 4.0, 400.0],
        "va_women_per_100k": [2.0, 3.0, 60.0, 120.0, 25.0, 40.0, 30.0, 20.0, 70.0, 80.0],
        "military_spend_per_capita": [600, 500, 200, 250, 300, 150, 700, 50, 100, 20],
        "displaced_per_100k": [0.0, 0.0, 50.0, 300.0, 20000.0, 40000.0, 50.0, 10.0, 500.0, 50000.0],
    }
    df = pd.DataFrame(data)
    df["iso3"] = df["country"].apply(iso3_from_country)
    return df

# ------------------------- Funções de download (ganchos) -------------------------
# NOTA: URLs e endpoints podem exigir ajuste. As funções retornam DataFrames ou lançam exceção.

def download_unodc_homicide(csv_url: str) -> pd.DataFrame:
    """Exemplo de função que baixa a tabela de homicídios da UNODC se fornecida uma URL pública."""
    r = requests.get(csv_url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def download_ucdp_battle_deaths(csv_url: str) -> pd.DataFrame:
    r = requests.get(csv_url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def download_sipri_military_spend(csv_url: str) -> pd.DataFrame:
    r = requests.get(csv_url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def download_unhcr_displaced(csv_url: str) -> pd.DataFrame:
    r = requests.get(csv_url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

# ACLED: requires token for full API access; deixamos o gancho

def download_acled_events(api_token: str, country: str = None, year: int = None) -> pd.DataFrame:
    """Exemplo mínimo: usar o endpoint REST da ACLED com token. O uso real exige leitura da doc da ACLED."""
    raise NotImplementedError("ACLED download placeholder - insira sua implementação com token")

# ------------------------- Pipeline de harmonização & cálculo -------------------------

def harmonize_and_compute_ip(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    # Certifique-se de que as colunas essenciais existem
    required = ["country", "homicide_per_100k", "battle_deaths_per_100k", "conflict_event_rate",
                "va_women_per_100k", "military_spend_per_capita", "displaced_per_100k"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Coluna obrigatória ausente: {c}")

    # Normalização min-max (1 = pior)
    norm = pd.DataFrame(index=df.index)
    for col in weights.keys():
        norm[col + "_norm"] = minmax_worse_is_high(df[col].astype(float))

    V = np.zeros(len(df))
    for col, w in weights.items():
        V = V + norm[col + "_norm"].values * w

    df = pd.concat([df, norm], axis=1)
    df["V_score"] = V
    df["IP_0_100"] = 100 * (1 - df["V_score"])  # 100 = mais pacífico
    return df

# ------------------------- Streamlit UI -------------------------

st.sidebar.header("Modo e fontes")
mode = st.sidebar.selectbox("Modo de execução:", ["DEMO (dados sintéticos)", "REAL (usar downloads)"])

st.sidebar.header("Pesos do índice (soma = 1)")
w_homicide = st.sidebar.slider("Homicídios (peso)", 0.0, 1.0, 0.25, 0.01)
w_battle = st.sidebar.slider("Mortes por conflito (peso)", 0.0, 1.0, 0.25, 0.01)
w_events = st.sidebar.slider("Eventos (peso)", 0.0, 1.0, 0.20, 0.01)
w_va = st.sidebar.slider("Violência contra mulheres (peso)", 0.0, 1.0, 0.15, 0.01)
w_mil = st.sidebar.slider("Gasto militar (peso)", 0.0, 1.0, 0.10, 0.01)
w_disp = st.sidebar.slider("Deslocados (peso)", 0.0, 1.0, 0.05, 0.01)

# Normalizar pesos para somarem 1 (se todos zero, usar default)
weights_raw = {
    "homicide_per_100k": w_homicide,
    "battle_deaths_per_100k": w_battle,
    "conflict_event_rate": w_events,
    "va_women_per_100k": w_va,
    "military_spend_per_capita": w_mil,
    "displaced_per_100k": w_disp,
}

if sum(weights_raw.values()) == 0:
    st.sidebar.warning("Todos os pesos são zero — usando pesos padrão.")
    weights = {k: v for k, v in {
        "homicide_per_100k": 0.25,
        "battle_deaths_per_100k": 0.25,
        "conflict_event_rate": 0.20,
        "va_women_per_100k": 0.15,
        "military_spend_per_capita": 0.10,
        "displaced_per_100k": 0.05,
    }.items()}
else:
    total = sum(weights_raw.values())
    weights = {k: v / total for k, v in weights_raw.items()}

st.sidebar.markdown("---")
st.sidebar.markdown("**Opções de exportação**")
export_csv = st.sidebar.checkbox("Exportar CSV resultante", value=True)

# Botões de ação
if st.sidebar.button("Executar pipeline"):
    try:
        if mode == "DEMO (dados sintéticos)":
            df = sample_data()
            st.success("Dados sintéticos carregados.")
        else:
            st.info("Tentando baixar dados reais — verifique sua conexão e URLs nas funções de download.")
            # Exemplo: o usuário deverá editar as URLs abaixo com as fontes reais
            try:
                # Exemplos de placeholders — editar conforme necessário
                unodc_url = st.sidebar.text_input("UNODC homicide CSV URL:", value="")
                ucdp_url = st.sidebar.text_input("UCDP battle deaths CSV URL:", value="")
                sipri_url = st.sidebar.text_input("SIPRI military spend CSV URL:", value="")
                unhcr_url = st.sidebar.text_input("UNHCR displaced CSV URL:", value="")

                # Simples validação: se qualquer campo vazio, aborta e sugere usar modo DEMO
                if any([not unodc_url, not ucdp_url, not sipri_url, not unhcr_url]):
                    st.error("Para usar o modo REAL, preencha as URLs nas opções da barra lateral ou escolha DEMO.")
                    st.stop()

                # Baixar — dependendo do formato da tabela, será necessário adaptar o parsing
                unodc_df = download_unodc_homicide(unodc_url)
                ucdp_df = download_ucdp_battle_deaths(ucdp_url)
                sipri_df = download_sipri_military_spend(sipri_url)
                unhcr_df = download_unhcr_displaced(unhcr_url)

                # TODO: harmonizar colunas e extrair as métricas desejadas por país
                st.success("Downloads concluídos — implemente a harmonização conforme sua fonte.")
                st.info("No momento, para demonstração, carregando dados sintéticos enquanto a harmonização não for implementada.")
                df = sample_data()

            except Exception as e:
                st.error(f"Falha ao baixar dados reais: {e}")
                st.info("Carregando dados sintéticos como fallback.")
                df = sample_data()

        # Calcular IP
        result = harmonize_and_compute_ip(df.copy(), weights)

        # Exibir resultados
        st.subheader("Resultado — Indicador de Paz (IP)")
        st.dataframe(result[["country", "iso3", "V_score", "IP_0_100"]].sort_values("IP_0_100", ascending=False))

        # Gráfico de barras
        fig = px.bar(result.sort_values("IP_0_100", ascending=False), x="country", y="IP_0_100",
                     labels={"IP_0_100": "Indicador de Paz (0-100)"}, title="Indicador de Paz por País")
        st.plotly_chart(fig, use_container_width=True)

        # Mapa coroplético (requer iso3). Se existirem iso3 válidos, desenhar mapa
        if result["iso3"].notnull().sum() > 0:
            map_df = result.dropna(subset=["iso3"]).copy()
            choropleth = px.choropleth(map_df, locations="iso3", color="IP_0_100",
                                       color_continuous_scale=px.colors.sequential.Plasma,
                                       title="Mapa — Indicador de Paz (ISO3)")
            st.plotly_chart(choropleth, use_container_width=True)
        else:
            st.info("Códigos ISO3 ausentes — não foi possível desenhar o mapa.")

        # Exportar CSV
        if export_csv:
            out_path = os.path.join(os.getcwd(), "indicador_de_paz_result.csv")
            result.to_csv(out_path, index=False)
            st.success(f"CSV exportado: {out_path}")

    except Exception as e:
        st.error(f"Erro no pipeline: {e}")

# Pequena explicação metodológica
st.sidebar.markdown("---")
st.sidebar.markdown("**Metodologia (resumo)**")
st.sidebar.markdown(
    "Normalização: min-max (0-1) onde 1 = pior; Agregação: soma ponderada; Indicador de Paz = 100 * (1 - V). Ajuste pesos conforme desejado.")

st.markdown("---")
st.markdown("### Dicas de personalização e próximos passos")
st.markdown("1. Substitua as funções de download por chamadas às APIs oficiais (UCDP, ACLED, UNODC, SIPRI, UNHCR).\n"
            "2. Harmonize nomes de países por ISO3 e calcule taxas per capita onde necessário.\n"
            "3. Implemente janelas móveis (3 anos) e tratamento de dados faltantes.\n"
            "4. Para produção, versione os dados e publique o código em um repositório (ex.: GitHub).")

st.markdown("---")
st.markdown("Se quiser, posso:")
st.markdown("- Preparar um repositório GitHub com este app e CI simples;\n- Executar o estudo de caso com dados reais para 10 países (eu mesmo busco e processo os dados);\n- Modularizar o projeto em pacotes conforme a opção A2.")


# Fim do arquivo
