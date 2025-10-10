import streamlit as st
import pandas as pd
import plotly.express as px

def reset_data():
    st.session_state['df'] = None

if 'df' not in st.session_state:
    st.session_state['df'] = None

pages = {
    "Página Principal": "main",
    "Análise Exploratória de Dados": "geostats",
    "Análise de Mudança de Modelo": "model_change"
}

selected_page = st.sidebar.selectbox("Escolha a página", list(pages.keys()))

if selected_page == "Página Principal":
    st.header("Upload de modelo")
    st.write("Estatísticas descritivas e visualização de dados.")

    model_1 = st.file_uploader("Faça upload do arquivo .csv", type="csv")
    if model_1 is not None:
        st.session_state['df'] = pd.read_csv(model_1)

    if st.button("Resetar dados"):
        reset_data()

    if st.session_state['df'] is not None:
        st.success("Arquivo carregado com sucesso!")
        st.write("Prévia dos dados:")
        st.write(st.session_state['df'])
        st.write("Estatísticas descritivas:")
        st.write(st.session_state['df'].describe())

elif selected_page == "Análise Exploratória de Dados":
    st.header("Análise Exploratória de Dados")

    # Verifica se o DataFrame está carregado
    if st.session_state['df'] is not None:
        df = st.session_state['df']

        # Layout para filtros e seleção de variável
        col1, col2 = st.columns([1, 1])

        # Identifica variáveis discretas (alfanuméricas)
        variaveis_discretas = df.select_dtypes(include=['object', 'category']).columns.tolist()

        with col1:
            # Filtro por variável discreta
            var_discreta = None
            valor_escolhido = None
            if variaveis_discretas:
                var_discreta = st.selectbox(
                    "Filtrar por variável discreta:",
                    ["Nenhum filtro"] + variaveis_discretas,
                    key="var_discreta_select",
                    help="Selecione uma variável discreta para filtrar os dados."
                )
                if var_discreta != "Nenhum filtro":
                    valores = df[var_discreta].unique().tolist()
                    valor_escolhido = st.selectbox(
                        f"Valor de {var_discreta}:",
                        valores,
                        key="valor_discreto_select",
                        help="Selecione o valor para filtrar a variável discreta."
                    )

        with col2:
            # Seleção da variável numérica para análise
            colunas_numericas = df.select_dtypes(include=['number']).columns
            coluna = st.selectbox(
                "Variável para análise:",
                colunas_numericas,
                key="coluna_analise_select",
                help="Selecione a variável numérica para análise."
            )

        # Aplica filtro se selecionado
        if var_discreta and var_discreta != "Nenhum filtro" and valor_escolhido is not None:
            df = df[df[var_discreta] == valor_escolhido]
            st.info(f"Filtro aplicado: {var_discreta} = {valor_escolhido}")

        st.write(f"**Variável selecionada:** `{coluna}`")
        st.write("Prévia dos dados filtrados:")
        st.dataframe(df.head(), use_container_width=True)

        # Dashboard de gráficos
        

        dash_col1, dash_col2 = st.columns(2)

        with dash_col1:
            st.subheader("Histograma")
            fig1 = px.histogram(df, x=coluna, nbins=30, opacity=0.7, title=f"Histograma de {coluna}")
            fig1.update_layout(xaxis_title=coluna, yaxis_title="Frequência", bargap=0.1)
            st.plotly_chart(fig1, use_container_width=True)

        with dash_col2:
            st.subheader("Boxplot")
            fig2 = px.box(df, x=coluna, points=False, title=f"Boxplot de {coluna}")
            fig2.update_layout(xaxis_title=coluna, yaxis_title=coluna)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Faça upload do arquivo .csv na página principal para começar.")

elif selected_page == "Análise de Mudança de Modelo":
    st.header("Análise de Mudança de Modelo")
    st.write("Conteúdo para análise de mudança entre modelos.")


