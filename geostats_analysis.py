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
        df = st.session_state['df']
        st.success("Arquivo carregado com sucesso!")
        
        # Verifica se existem as colunas de coordenadas
        coord_columns = ['XC', 'YC', 'ZC']
        has_coordinates = all(col in df.columns for col in coord_columns)
        
        if has_coordinates:
            st.subheader("Visualização 3D dos Pontos")
            
            # Criar scatter 3D
            fig_3d = px.scatter_3d(
                df,
                x='XC',
                y='YC',
                z='ZC',
                title="Distribuição Espacial dos Pontos",
                labels={'XC': 'Coordenada X', 'YC': 'Coordenada Y', 'ZC': 'Coordenada Z'}
            )
            
            # Ajustar layout para melhor visualização
            fig_3d.update_layout(
                scene=dict(
                    aspectmode='data',  # preserva a escala real dos dados
                    camera=dict(
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0),
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                height=600
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # Mostrar estatísticas das coordenadas
            st.subheader("Estatísticas das Coordenadas")
            st.write(df[coord_columns].describe())
        
        st.subheader("Prévia dos dados")
        st.write(df)
        
        st.subheader("Estatísticas descritivas gerais")
        st.write(df.describe())

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
        dash_col1, dash_col2, dash_col3 = st.columns(3)

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

        with dash_col3:
            st.subheader("Crossplot")
            # Seleção da segunda variável para o crossplot
            coluna_y = st.selectbox(
                "Variável Y para crossplot:",
                [col for col in colunas_numericas if col != coluna],
                key="coluna_crossplot_select"
            )
            
            # Opções de transformação
            col_options1, col_options2 = st.columns(2)
            with col_options1:
                use_log_x = st.checkbox("Log10 escala X", key="log_x")
                same_bounds = st.checkbox("Mesmos limites nos eixos", key="same_bounds")
            with col_options2:
                use_log_y = st.checkbox("Log10 escala Y", key="log_y")
                swap_vars = st.checkbox("Trocar variáveis", key="swap_vars")

            # Preparar dados para o crossplot
            plot_data = df.copy()
            x_var = coluna_y if swap_vars else coluna
            y_var = coluna if swap_vars else coluna_y

            if use_log_x:
                plot_data = plot_data[plot_data[x_var] > 0]
            if use_log_y:
                plot_data = plot_data[plot_data[y_var] > 0]

            # Criar crossplot
            fig3 = px.scatter(
                plot_data,
                x=x_var,
                y=y_var,
                title=f"Crossplot: {x_var} vs {y_var}",
                trendline="ols"
            )

            # Aplicar transformações log se selecionadas
            if use_log_x:
                fig3.update_xaxes(type="log")
            if use_log_y:
                fig3.update_yaxes(type="log")

            # Aplicar mesmos limites se selecionado
            if same_bounds:
                all_values = pd.concat([plot_data[x_var], plot_data[y_var]])
                min_val = all_values.min()
                max_val = all_values.max()
                fig3.update_xaxes(range=[min_val, max_val])
                fig3.update_yaxes(range=[min_val, max_val])

            # Calcular e mostrar correlação
            corr = plot_data[x_var].corr(plot_data[y_var])
            st.write(f"Correlação: {corr:.3f}")
            
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("Faça upload do arquivo .csv na página principal para começar.")

elif selected_page == "Análise de Mudança de Modelo":
    st.header("Análise de Mudança de Modelo")
    st.write("Conteúdo para análise de mudança entre modelos.")


